#pragma once
#include <omp.h>
#include <immintrin.h>
#include "utils.h"
#include "parameters.h"

#if ISAVX
#define SET8F(a) _mm256_set1_ps(a)
#define SET8FE(a, b, c, d, e, f, g, i) _mm256_set_ps(a, b, c, d, e, f, g, i)
#define MUL8F(a, b) _mm256_mul_ps(a, b)
#define DIV8F(a, b) _mm256_div_ps(a, b)
#define ADD8F(a, b) _mm256_add_ps(a, b)
#define SUB8F(a, b) _mm256_sub_ps(a, b)
#define LOG28F(a) _mm256_log2_ps(a)
#define GETMASK(a, b) _mm256_cvtps_epi32(_mm256_cmp_ps(a, b, _CMP_LT_OS));
#define MASKLOAD(a, b) _mm256_maskload_ps(a, b)

// Reference: https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
static float sum8f(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
#endif

class EntropyBasedThreshold 
{
public:
	EntropyBasedThreshold(cv::Mat& src);
	int Process();

private:
	uchar* _data;
	float* _glcm;
	int _width;
	int _height;

	void _getGrayLevelCoOccurrenceMatrix();
	int _entropyThesholding();
#if ISAVX 
	inline void _loadPixel(__m256& x, __m256& p, int width, int r, int c, int cBoundary);
#endif
};