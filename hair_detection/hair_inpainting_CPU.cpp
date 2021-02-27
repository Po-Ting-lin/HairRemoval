#include "hair_inpainting_CPU.h"

void normalizeImage(cv::Mat& srcImage, cv::Mat& srcMask, float* dstImage, float* dstMask, float* dstMaskImage, bool channelSplit) {
	const int width = srcImage.cols;
	const int height = srcImage.rows;
	uchar* src_image_ptr = srcImage.data;
	uchar* src_mask_ptr = srcMask.data;
	for (int i = 0; i < height * width; i++) {
		dstMask[i] = src_mask_ptr[i] != 0 ? 0.0f : 1.0f;
	}
	int max_r = 0;
	int max_g = 0;
	int max_b = 0;
	int min_r = 255;
	int min_g = 255;
	int min_b = 255;
	int pixel = 0;
	int range_r, range_g, range_b;
	int i = 0;
	int maskI = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			i = y * (width * 3) + (x * 3);
			pixel = src_image_ptr[i];
			if (pixel > max_r) max_r = pixel;
			if (pixel < min_r) min_r = pixel;
			pixel = src_image_ptr[i + 1];
			if (pixel > max_g) max_g = pixel;
			if (pixel < min_g) min_g = pixel;
			pixel = src_image_ptr[i + 2];
			if (pixel > max_b) max_b = pixel;
			if (pixel < min_b) min_b = pixel;
		}
	}
	range_r = max_r - min_r;
	range_g = max_g - min_g;
	range_b = max_b - min_b;
	if (channelSplit) {
		int min_list[] = { min_r, min_g, min_b };
		int range_list[] = { range_r, range_g, range_b };
		int srcI = 0;
		int dstI = 0;
		for (int k = 0; k < 3; k++) {
			int channel_offset = k * width * height;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					maskI = y * width + x;
					srcI = y * (width * 3) + (x * 3) + k;
					dstI = channel_offset + maskI;
					float value = ((float)src_image_ptr[srcI] - min_list[k]) / range_list[k];
					dstImage[dstI] = value;
					dstMaskImage[dstI] = dstMask[maskI] > 0.0f ? value : 1.0f;
				}
			}
		}
	}
	else {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				i = y * (width * 3) + (x * 3);
				maskI = y * width + x;
				float r = ((float)src_image_ptr[i] - min_r) / range_r;
				float g = ((float)src_image_ptr[i + 1] - min_g) / range_g;
				float b = ((float)src_image_ptr[i + 2] - min_b) / range_b;
				dstImage[i] = r;
				dstImage[i + 1] = g;
				dstImage[i + 2] = b;
				dstMaskImage[i] = dstMask[maskI] > 0.0f ? r : 1.0f;
				dstMaskImage[i + 1] = dstMask[maskI] > 0.0f ? g : 1.0f;
				dstMaskImage[i + 2] = dstMask[maskI] > 0.0f ? b : 1.0f;
			}
		}
	}
}

void mergeChannels(float* srcImage, float* dstImage, HairInpaintInfo info) {
	int srcI = 0;
	int dstI = 0;
	float temp = 0;
	for (int k = 0; k < 3; k++) {
		int channel_offset = k * info.Width * info.Height;
		for (int y = 0; y < info.Height; y++) {
			for (int x = 0; x < info.Width; x++) {
				dstI = y * (info.Width * info.Channels) + (x * info.Channels) + k;
				srcI = channel_offset + y * info.Width + x;
				dstImage[dstI] = srcImage[srcI];
			}
		}
	}
}

void hairInpaintingCPU(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, HairInpaintInfo info) {
	cv::resize(src, src, cv::Size(info.Width, info.Height));
	cv::resize(mask, mask, cv::Size(info.Width, info.Height));
	
	float* normalized_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	float* normalized_mask = (float*)malloc(info.NumberOfC1Elements * sizeof(float));
	float* normalized_masked_src = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	float* img_u = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	float* img_u_temp = (float*)malloc(info.NumberOfC3Elements * sizeof(float));
	normalizeImage(src, mask, normalized_src, normalized_mask, normalized_masked_src, true);
	memcpy(img_u, normalized_masked_src, info.NumberOfC3Elements * sizeof(float));
	memcpy(img_u_temp, normalized_masked_src, info.NumberOfC3Elements * sizeof(float));

	const float dt = 0.1f;
	const float cw = 4.0f;
	int x_boundary = info.Width - 1;

	for (int i = 0; i < 500; i++) {
		for (int k = 0; k < info.Channels; k++) {
			int channel_offset = k * info.Width * info.Height;
#pragma omp parallel for
			for (int y = 1; y < info.Height - 1; y++) {
#if ISAVX
				__m256 _pA = SET8F(0.0f);
				__m256 _pC = SET8F(0.0f);
				__m256 _mA = SET8F(0.0f);
				__m256 _mC = SET8F(0.0f);
				__m256 _eA = SET8F(0.0f);
				__m256 _eC = SET8F(0.0f);
				__m256 _dt = SET8F(dt);
				__m256 _cw = SET8F(cw);
				__m256 _x;
				__m256 _c, _u, _d, _l, _r, _mc, _oc;
				__m256i _x_mask;
				for (int x = 1; x < x_boundary; x += 8) {
					int c1i = y * info.Width + x;
					int c3i = channel_offset + c1i;
					int c3ui = channel_offset + (y - 1) * info.Width + x;
					int c3di = channel_offset + (y + 1) * info.Width + x;
					int c3li = channel_offset + y * info.Width + (x - 1);
					int c3ri = channel_offset + y * info.Width + (x + 1);

					_x = SET8FE(x + 7.0f, x + 6.0f, x + 5.0f, x + 4.0f, x + 3.0f, x + 2.0f, x + 1.0f, x);
					_x_mask = GETMASK(_x, SET8F(x_boundary));

					_c = SET8F(0.0f);
					_u = SET8F(0.0f);
					_d = SET8F(0.0f);
					_l = SET8F(0.0f);
					_r = SET8F(0.0f);
					_mc = SET8F(0.0f);
					_oc = SET8F(0.0f);
					_c = MASKLOAD(&img_u[c3i], _x_mask);
					_u = MASKLOAD(&img_u[c3ui], _x_mask);
					_d = MASKLOAD(&img_u[c3di], _x_mask);
					_l = MASKLOAD(&img_u[c3li], _x_mask);
					_r = MASKLOAD(&img_u[c3ri], _x_mask);
					_mc = MASKLOAD(&normalized_mask[c1i], _x_mask);
					_oc = MASKLOAD(&normalized_masked_src[c3i], _x_mask);
					MASKSTORE(&img_u[c3i]
						, _x_mask
						, SUB8F(ADD8F(_c, MUL8F(_dt, SUB8F(ADD8F(_u, ADD8F(_d, ADD8F(_l, _r))), MUL8F(_cw, _c)))), MUL8F(_dt, MUL8F(_mc, SUB8F(_c, _oc)))));
				}
#else
				for (int x = 1; x < x_boundary; x++) {
					int c1i = y * info.Width + x;
					int c3i = channel_offset + c1i;
					int c3ui = channel_offset + (y - 1) * info.Width + x;
					int c3di = channel_offset + (y + 1) * info.Width + x;
					int c3li = channel_offset + y * info.Width + (x - 1);
					int c3ri = channel_offset + y * info.Width + (x + 1);

					img_u[c3i] = img_u[c3i]
						+ dt * (img_u[c3ui] + img_u[c3di] + img_u[c3li] + img_u[c3ri] - 4.0f * img_u[c3i])
						- dt * normalized_mask[c1i] * (img_u[c3i] - normalized_masked_src[c3i]);
				}
#endif
			}
		}
	}
	memcpy(img_u_temp, img_u, info.NumberOfC3Elements * sizeof(float));
	mergeChannels(img_u_temp, img_u, info);
	cv::Mat dst_mat(info.Height, info.Width, CV_32FC3, img_u);
	cv::resize(dst_mat, dst_mat, cv::Size(info.Width * info.Rescale, info.Height * info.Rescale));
	dst = dst_mat;
}