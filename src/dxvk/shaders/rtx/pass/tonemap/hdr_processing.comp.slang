/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx/pass/tonemap/tonemapping.slangh"
#include "rtx/pass/tonemap/tonemapping_noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/noise.slangh"

// HDR Processing binding indices are now defined in tonemapping.h

layout(binding = HDR_PROCESSING_BLUE_NOISE_TEXTURE)
Texture2DArray BlueNoise;

layout(binding = HDR_PROCESSING_INPUT_BUFFER)
RWTexture2D<float4> InColorBuffer;

layout(binding = HDR_PROCESSING_OUTPUT_BUFFER)
RWTexture2D<float4> OutColorBuffer;

layout(binding = HDR_PROCESSING_EXPOSURE_INPUT)
RWTexture1D<float> InExposure;

// HDR Processing Arguments
struct HDRProcessingArgs {
  uint enableAutoExposure;
  float hdrMaxLuminance;
  float hdrMinLuminance;
  float hdrPaperWhiteLuminance;
  float exposureFactor;
  uint frameIndex;
  uint hdrFormat;  // 0=Linear, 1=PQ, 2=HLG
  float hdrExposureBias;
  float hdrBrightness;
  uint hdrToneMapper;  // 0=None, 1=ACES_HDR
  uint hdrEnableDithering;
  float hdrShadows;
  float hdrMidtones;
  float hdrHighlights;
  float hdrBlueNoiseAmplitude;
};

layout(push_constant)
ConstantBuffer<HDRProcessingArgs> cb;

// Convert from Rec.709 to Rec.2020 color space for HDR
float3 rec709ToRec2020(float3 color) 
{
  // Conversion matrix from Rec.709 to Rec.2020
  // These are the official ITU-R BT.2020 conversion matrices
  const float3x3 conversionMatrix = float3x3(
    0.6274040f,  0.3292820f,  0.0433136f,
    0.0690970f,  0.9195400f,  0.0113612f,
    0.0163916f,  0.0880132f,  0.8955950f
  );
  
  return mul(conversionMatrix, color);
}

// Convert from linear RGB to PQ (Perceptual Quantizer) for HDR10
float3 linearToPQ(float3 color) 
{
  // PQ constants for HDR10 (ST.2084)
  const float m1 = 0.1593017578125f; // 2610.0 / 16384.0
  const float m2 = 78.84375f; // 2523.0 / 32.0
  const float c1 = 0.8359375f; // 3424.0 / 4096.0
  const float c2 = 18.8515625f; // 2413.0 / 128.0
  const float c3 = 18.6875f; // 2392.0 / 128.0
  
  // Clamp input to valid range [0, 1] for PQ
  color = clamp(color, float3(0.0f), float3(1.0f));
  
  // Apply PQ curve per channel with safety checks
  float3 result;
  for (int i = 0; i < 3; ++i) {
    float channel = color[i];
    if (channel <= 0.0f) {
      result[i] = 0.0f;
    } else {
      float powM1 = pow(channel, m1);
      result[i] = pow((c1 + c2 * powM1) / (1.0f + c3 * powM1), m2);
    }
  }
  
  return result;
}

// Convert from linear RGB to HLG (Hybrid Log-Gamma) for HDR
float3 linearToHLG(float3 color) 
{
  // HLG constants (ITU-R BT.2100)
  const float a = 0.17883277f;
  const float b = 0.28466892f; // 1.0 - 4.0 * a
  const float c = 0.55991073f; // 0.5 - a * ln(4.0 * a)
  const float gamma = 1.2f;
  const float threshold = 1.0f / 12.0f;
  
  // Clamp input to valid range [0, 1] for HLG
  color = clamp(color, float3(0.0f), float3(1.0f));
  
  float3 result;
  for (int i = 0; i < 3; ++i) {
    float channel = color[i];
    if (channel <= 0.0f) {
      result[i] = 0.0f;
    } else if (channel <= threshold) {
      result[i] = sqrt(3.0f * channel);
    } else {
      result[i] = a * log(12.0f * channel - b) + c;
    }
  }
  
  // Apply system gamma with safety check
  return pow(max(result, float3(0.0f)), 1.0f / gamma);
}

// Enhanced color space conversion with HDR10 and HLG support
float3 convertColorSpace(float3 color, bool usePQ, bool useHLG) 
{
  // Convert to target gamut only if using HDR transfer functions (BT.2020 gamut)
  if (usePQ || useHLG) {
    color = rec709ToRec2020(color);
  }
  
  // Apply appropriate transfer function
  if (usePQ) {
    // For PQ: Input colors are assumed to be in relative [0-1] range where 1.0 = paper white
    // Scale to nits first: multiply by paper white to get actual nits, then clamp to display range
    float3 colorInNits = color * cb.hdrPaperWhiteLuminance;
    colorInNits = clamp(colorInNits, float3(cb.hdrMinLuminance), float3(cb.hdrMaxLuminance));
    // Then normalize by PQ reference peak (10,000 nits) for PQ encoding
    const float PQ_MAX_NITS = 10000.0f;
    float3 normalizedColor = colorInNits / PQ_MAX_NITS;
    return linearToPQ(normalizedColor);
  } else if (useHLG) {
    // For HLG: Input colors are assumed to be in relative [0-1] range where 1.0 = paper white
    // Scale to the display peak range for HLG and clamp to display range
    float3 colorInNits = color * cb.hdrPaperWhiteLuminance;
    colorInNits = clamp(colorInNits, float3(cb.hdrMinLuminance), float3(cb.hdrMaxLuminance));
    // HLG system gamma is designed for [0,1] input where 1.0 = display peak
    float3 normalizedColor = colorInNits / cb.hdrMaxLuminance;
    return linearToHLG(normalizedColor);
  } else {
    // Linear/scRGB: keep BT.709 linear without gamut conversion
    return color;
  }
}

// Note: Old dithering functions removed - now using high-quality blue noise dithering

// Note: Complex Bayer pattern dithering functions removed - using superior blue noise approach

// Note: Removed complex PQ-specific dithering - blue noise handles this better

// Note: All complex manual dithering removed - using professional blue noise approach

// Note: Pre-tone-mapping dithering removed - blue noise handles this optimally

// Precision-enhanced ACES modified for HDR range preservation
float3 acesHDR(float3 color, float maxLuminance, float paperWhite)
{
  // Use the exact same matrices as the working ACES implementation
  const float3x3 aces_input_matrix = float3x3(
    0.59719f, 0.35458f, 0.04823f,
    0.07600f, 0.90834f, 0.01566f,
    0.02840f, 0.13383f, 0.83777f
  );
  
  const float3x3 aces_output_matrix = float3x3(
    1.60475f, -0.53108f, -0.07367f,
    -0.10208f, 1.10813f, -0.00605f,
    -0.00327f, -0.07276f, 1.07602f
  );
  
  // Apply input matrix transform
  color = mul(aces_input_matrix, color);
  
  // Apply RRT and ODT (same as working implementation)
  float3 a = color * (color + 0.0245786f) - 0.000090537f;
  float3 b = color * (0.983729f * color + 0.4329510f) + 0.238081f;
  color = a / b;
  
  // Apply output matrix transform
  color = mul(aces_output_matrix, color);
  
  // Clamp to valid range - ACES already outputs in proper range for HDR
  // Don't scale by paper white here as it will cause dimming
  return saturate(color);
}

// Shadow/Midtone/Highlight adjustments (log-luminance tri-band with smooth masks)
float3 applyShadowMidtoneHighlight(float3 color, float shadows, float midtones, float highlights)
{
  // Perceptual luminance
  const float eps = 1e-6f;
  float luma = calcBt709Luminance(color);
  float logLuma = log2(max(luma, eps));

  // Map log-luma from a reasonable HDR range [minStops, maxStops] to [0,1]
  // These boundaries roughly cover very dark shadows to bright highlights in relative space
  const float minStops = -10.0f; // extend shadow range
  const float maxStops =   8.0f; // extend highlight range to catch speculars
  float l01 = saturate((logLuma - minStops) / (maxStops - minStops));

  // Smooth tri-band masks with limited overlap
  // Tuned to give the highlights control more presence in typical content
  // - Shadows: strong below ~0.45
  // - Highlights: start earlier (~0.55) to ensure visible impact
  // - Midtones: fills remaining energy between the two
  float shadowsMask    = 1.0f - smoothstep(0.20f, 0.45f, l01);
  float highlightsMask =        smoothstep(0.55f, 0.75f, l01);
  float midtonesMask   = saturate(1.0f - shadowsMask - highlightsMask);

  // Normalize so masks sum to 1 where present to reduce unintended global shifts
  float maskSum = shadowsMask + midtonesMask + highlightsMask;
  if (maskSum > 0.0f) {
    float inv = rcp(maskSum);
    shadowsMask    *= inv;
    midtonesMask   *= inv;
    highlightsMask *= inv;
  }

  // Combine adjustments in EV space with the masks
  // Slightly bias highlights to improve perceptual response in narrow bright regions
  float ev = shadows    * shadowsMask
           + midtones   * midtonesMask
           + highlights * (highlightsMask * 1.15f);

  return color * exp2(ev);
}

// Main HDR tone mapping dispatch
float3 hdrToneMapping(float3 color, float maxLuminance, float paperWhite)
{
  // Apply exposure adjustment first (unless bypassing everything)
  if (cb.hdrToneMapper != 0) {
    color *= getExposure(InExposure, cb.enableAutoExposure != 0, cb.exposureFactor);
  }
  
  // Apply selected HDR tone mapping method
  switch (cb.hdrToneMapper) {
    case 0: // None - true bypass for debugging
      // Only apply minimal exposure if auto exposure is enabled
      if (cb.enableAutoExposure != 0) {
        color *= getExposure(InExposure, true, 1.0f);
      }
      return color; // No clamping to preserve original precision
    case 1: // ACES_HDR
      return acesHDR(color, maxLuminance, paperWhite);
    default:
      return color; // Default to no tone mapping if an unknown case is selected
  }
}

// Main HDR processing function
float3 processHDR(float3 inputColor, uint2 pixelPosition)
{
  float3 color = inputColor;
  
  // Note: Pre-tone mapping dithering removed - now using high-quality blue noise dithering after processing
  
  // Apply HDR tone mapping (relative domain)
  color = hdrToneMapping(color, cb.hdrMaxLuminance, cb.hdrPaperWhiteLuminance);
  
  // Apply HDR-specific brightness and exposure adjustments (relative domain)
  // Apply exposure bias (in EV stops)
  color *= exp2(cb.hdrExposureBias);
  
  // Apply brightness multiplier
  color *= cb.hdrBrightness;
  
  // Apply shadow/midtone/highlight adjustments
  color = applyShadowMidtoneHighlight(color, cb.hdrShadows, cb.hdrMidtones, cb.hdrHighlights);
  
  // Note: no clamping here; clamping is applied when converting to nits per format
  
  // Enhanced color space conversion with HDR format support
  bool usePQ = (cb.hdrFormat == 1);  // PQ/HDR10
  bool useHLG = (cb.hdrFormat == 2); // HLG
  color = convertColorSpace(color, usePQ, useHLG);
  
  // Apply high-quality blue noise dithering AFTER color space conversion
  // This uses the same blue noise infrastructure as SDR for consistent quality
  color = ditherToHDR(
    color,
    pixelPosition,
    cb.frameIndex,
    cb.hdrEnableDithering != 0,
    true, // Enable temporal dithering for smoothest results
    cb.hdrFormat,
    cb.hdrBlueNoiseAmplitude
  );
  
  return color;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  float4 inputColor = imageLoad(InColorBuffer, threadId);
  
  float3 processedColor = processHDR(inputColor.rgb, threadId);
  
  // Preserve alpha channel
  OutColorBuffer[threadId] = float4(processedColor, inputColor.a);
} 