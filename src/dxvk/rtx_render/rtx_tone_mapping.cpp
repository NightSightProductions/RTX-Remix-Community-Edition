/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_tone_mapping.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx.h"
#include "rtx/pass/tonemap/tonemapping.h"

#include <rtx_shaders/auto_exposure.h>
#include <rtx_shaders/auto_exposure_histogram.h>
#include <rtx_shaders/tonemapping_histogram.h>
#include <rtx_shaders/tonemapping_tone_curve.h>
#include <rtx_shaders/tonemapping_apply_tonemapping.h>
#include <rtx_shaders/hdr_processing.h>
#include "rtx_imgui.h"
#include "rtx/utility/debug_view_indices.h"
#include <algorithm>
#include <cmath>
#include <vector>

static_assert((TONEMAPPING_TONE_CURVE_SAMPLE_COUNT & 1) == 0, "The shader expects a sample count that is a multiple of 2.");

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class HistogramShader : public ManagedShader
    {
      SHADER_SOURCE(HistogramShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_histogram)

      PUSH_CONSTANTS(ToneMappingHistogramArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE1D(TONEMAPPING_HISTOGRAM_HISTOGRAM_INPUT_OUTPUT)
        RW_TEXTURE2D_READONLY(TONEMAPPING_HISTOGRAM_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_HISTOGRAM_EXPOSURE_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(HistogramShader);

    class ToneCurveShader : public ManagedShader
    {
      SHADER_SOURCE(ToneCurveShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_tone_curve)

      PUSH_CONSTANTS(ToneMappingCurveArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE1D(TONEMAPPING_TONE_CURVE_HISTOGRAM_INPUT_OUTPUT)
        RW_TEXTURE1D(TONEMAPPING_TONE_CURVE_TONE_CURVE_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ToneCurveShader);

    class ApplyTonemappingShader : public ManagedShader
    {
      SHADER_SOURCE(ApplyTonemappingShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_apply_tonemapping)

      PUSH_CONSTANTS(ToneMappingApplyToneMappingArgs)

      BEGIN_PARAMETER()
        TEXTURE2DARRAY(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)
        SAMPLER1D(TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ApplyTonemappingShader);
  }

    class HDRProcessingShader : public ManagedShader
    {
      SHADER_SOURCE(HDRProcessingShader, VK_SHADER_STAGE_COMPUTE_BIT, hdr_processing)

      PUSH_CONSTANTS(HDRProcessingArgs)

      BEGIN_PARAMETER()
        TEXTURE2DARRAY(HDR_PROCESSING_BLUE_NOISE_TEXTURE)
        RW_TEXTURE2D(HDR_PROCESSING_INPUT_BUFFER)
        RW_TEXTURE2D(HDR_PROCESSING_OUTPUT_BUFFER)
        RW_TEXTURE1D_READONLY(HDR_PROCESSING_EXPOSURE_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(HDRProcessingShader);
  
  DxvkToneMapping::DxvkToneMapping(DxvkDevice* device)
  : CommonDeviceObject(device), m_vkd(device->vkd())  {
  }
  
  DxvkToneMapping::~DxvkToneMapping()  {  }

  void DxvkToneMapping::showImguiSettings() {

    ImGui::DragFloat("Global Exposure", &exposureBiasObject(), 0.01f, -4.f, 4.f);
    
    ImGui::Separator();
    
    ImGui::Checkbox("Color Grading Enabled", &colorGradingEnabledObject());
    if (colorGradingEnabled()) {
      ImGui::Indent();
      ImGui::DragFloat("Contrast", &contrastObject(), 0.01f, 0.f, 1.f);
      ImGui::DragFloat("Saturation", &saturationObject(), 0.01f, 0.f, 1.f);
      ImGui::DragFloat3("Color Balance", &colorBalanceObject(), 0.01f, 0.f, 1.f);
      ImGui::Separator();
      ImGui::Unindent();
    }
    
    // Curve Editor Section
    ImGui::Separator();
    ImGui::Text("Curve Editor");
    ImGui::Checkbox("Enable Curve Editor", &enableCurveEditorObject());
    
    if (enableCurveEditor()) {
      ImGui::Indent();
      ImGui::Checkbox("Use Custom Curve", &useCustomCurveObject());
      
      // Initialize curve points if not done
      if (!m_curveEditorInitialized) {
        m_customCurvePoints.clear();
        m_customCurvePoints.emplace_back(0.0f, 0.0f);   // Black point
        m_customCurvePoints.emplace_back(1.0f, 1.0f);   // White point
        m_curveEditorInitialized = true;
      }
      
      // Show curve editor widget
      if (showCurveEditor("Tone Curve", m_customCurvePoints, ImVec2(300, 300))) {
        // Curve was modified, mark as changed
        m_isCurveChanged = true;
      }
      
      ImGui::Text("Instructions:");
      ImGui::BulletText("Left click to add control points");
      ImGui::BulletText("Right click to remove control points");
      ImGui::BulletText("Drag points to adjust curve");
      ImGui::BulletText("X-axis: Input (shadows to highlights)");
      ImGui::BulletText("Y-axis: Output (dark to bright)");
      
      if (ImGui::Button("Reset Curve")) {
        m_customCurvePoints.clear();
        m_customCurvePoints.emplace_back(0.0f, 0.0f);
        m_customCurvePoints.emplace_back(1.0f, 1.0f);
        m_isCurveChanged = true;
      }
      
      ImGui::Unindent();
    }

    ImGui::Separator();

    ImGui::Checkbox("Tonemapping Enabled", &tonemappingEnabledObject());
    if (tonemappingEnabled()) {  // Show tonemapping options when enabled
      ImGui::Indent();
      
      // Tone mapping operator selection
      const char* operators[] = { "Standard", "ACES", "AgX" };
      int currentOp = useAgX() ? 2 : (finalizeWithACES() ? 1 : 0);
      if (ImGui::Combo("Tone Mapping Operator", &currentOp, operators, IM_ARRAYSIZE(operators))) {
        finalizeWithACES.setDeferred(currentOp == 1);
        useAgX.setDeferred(currentOp == 2);
      }

      // AgX-specific controls (only show when AgX is selected)
      if (useAgX()) {
        ImGui::Indent();
        ImGui::Text("AgX Controls:");
        ImGui::Separator();
        
        // Basic controls
        ImGui::DragFloat("AgX Gamma", &agxGammaObject(), 0.01f, 0.5f, 3.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat("AgX Saturation", &agxSaturationObject(), 0.01f, 0.5f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat("AgX Exposure Offset", &agxExposureOffsetObject(), 0.01f, -2.0f, 2.0f, "%.3f EV", ImGuiSliderFlags_AlwaysClamp);
        
        ImGui::Separator();
        
        // Look selection
        const char* looks[] = { "None", "Punchy", "Golden", "Greyscale" };
        ImGui::Combo("AgX Look", &agxLookObject(), looks, IM_ARRAYSIZE(looks));
        
        ImGui::Separator();
        
        // Advanced controls
        ImGui::Text("Advanced:");
        ImGui::DragFloat("AgX Contrast", &agxContrastObject(), 0.01f, 0.5f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat("AgX Slope", &agxSlopeObject(), 0.01f, 0.5f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat("AgX Power", &agxPowerObject(), 0.01f, 0.5f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        
        ImGui::Unindent();
      }

      ImGui::Combo("Dither Mode", &ditherModeObject(), "Disabled\0Spatial\0Spatial + Temporal\0");

      ImGui::Checkbox("Tuning Mode", &tuningModeObject());
      if (tuningMode()) {
        ImGui::Indent();

        ImGui::DragFloat("Curve Shift", &curveShiftObject(), 0.01f, 0.f, 0.f);
        ImGui::DragFloat("Shadow Min Slope", &shadowMinSlopeObject(), 0.01f, 0.f, 0.f);
        ImGui::DragFloat("Shadow Contrast", &shadowContrastObject(), 0.01f, 0.f, 0.f);
        ImGui::DragFloat("Shadow Contrast End", &shadowContrastEndObject(), 0.01f, 0.f, 0.f);
        ImGui::DragFloat("Min Stops", &toneCurveMinStopsObject(), 0.01f, 0.f, 0.f);
        ImGui::DragFloat("Max Stops", &toneCurveMaxStopsObject(), 0.01f, 0.f, 0.f);

        ImGui::DragFloat("Max Exposure Increase", &maxExposureIncreaseObject(), 0.01f, 0.f, 0.f);
        ImGui::DragFloat("Dynamic Range", &dynamicRangeObject(), 0.01f, 0.f, 0.f);

        ImGui::Unindent();
      }
      ImGui::Separator();
      ImGui::Unindent();
    }
  }

  void DxvkToneMapping::createResources(Rc<RtxContext> ctx) {
    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_1D;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.numLayers = 1;
    desc.mipLevels = 1;
    desc.stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_1D;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;
    viewInfo.format = desc.format;

    desc.extent = VkExtent3D{ TONEMAPPING_TONE_CURVE_SAMPLE_COUNT, 1, 1 };

    viewInfo.format = desc.format = VK_FORMAT_R32_UINT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_toneHistogram.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "tone mapper histogram");
    m_toneHistogram.view = device()->createImageView(m_toneHistogram.image, viewInfo);
    ctx->changeImageLayout(m_toneHistogram.image, VK_IMAGE_LAYOUT_GENERAL);

    viewInfo.format = desc.format = VK_FORMAT_R32_SFLOAT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    m_toneCurve.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "tone mapper curve");
    m_toneCurve.view = device()->createImageView(m_toneCurve.image, viewInfo);
    ctx->changeImageLayout(m_toneCurve.image, VK_IMAGE_LAYOUT_GENERAL);
  }

  void DxvkToneMapping::dispatchHistogram(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::Resource& colorBuffer,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Tonemap: Generate Histogram");

    // Clear the histogram resource
    if(m_resetState) {
      VkClearColorValue clearColor;
      clearColor.float32[0] = clearColor.float32[1] = clearColor.float32[2] = clearColor.float32[3] = 0;

      VkImageSubresourceRange subRange = {};
      subRange.layerCount = 1;
      subRange.levelCount = 1;
      subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

      ctx->clearColorImage(m_toneHistogram.image, clearColor, subRange);
    }

    // Prepare shader arguments
    ToneMappingHistogramArgs pushArgs = {};
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.toneCurveMinStops = toneCurveMinStops();
    pushArgs.toneCurveMaxStops = toneCurveMaxStops();
    pushArgs.exposureFactor = exp2f(exposureBias() + RtxOptions::calcUserEVBias());

    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

    VkExtent3D workgroups = util::computeBlockCount(colorBuffer.view->imageInfo().extent, VkExtent3D{16, 16, 1 });

    ctx->bindResourceView(TONEMAPPING_HISTOGRAM_COLOR_INPUT, colorBuffer.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_HISTOGRAM_HISTOGRAM_INPUT_OUTPUT, m_toneHistogram.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_HISTOGRAM_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, HistogramShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatchToneCurve(
    Rc<RtxContext> ctx) {

    ScopedGpuProfileZone(ctx, "Tonemap: Calculate Tone Curve");

    // Prepare shader arguments
    ToneMappingCurveArgs pushArgs = {};
    pushArgs.dynamicRange = dynamicRange();
    pushArgs.shadowMinSlope = shadowMinSlope();
    pushArgs.shadowContrast = shadowContrast();
    pushArgs.shadowContrastEnd = shadowContrastEnd();
    pushArgs.maxExposureIncrease = maxExposureIncrease();
    pushArgs.curveShift = curveShift();
    pushArgs.toneCurveMinStops = toneCurveMinStops();
    pushArgs.toneCurveMaxStops = toneCurveMaxStops();
    pushArgs.needsReset = m_resetState;

    VkExtent3D workgroups = VkExtent3D{ TONEMAPPING_TONE_CURVE_SAMPLE_COUNT, 1, 1 };

    ctx->bindResourceView(TONEMAPPING_TONE_CURVE_HISTOGRAM_INPUT_OUTPUT, m_toneHistogram.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_TONE_CURVE_TONE_CURVE_INPUT_OUTPUT, m_toneCurve.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ToneCurveShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatchApplyToneMapping(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::Resource& inputBuffer,
    const Resources::Resource& colorBuffer,
    bool performSRGBConversion,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Apply Tone Mapping");

    const VkExtent3D workgroups = util::computeBlockCount(colorBuffer.view->imageInfo().extent, VkExtent3D{ 16 , 16, 1 });

    // Prepare shader arguments
    ToneMappingApplyToneMappingArgs pushArgs = {};
    pushArgs.toneMappingEnabled = tonemappingEnabled();
    pushArgs.colorGradingEnabled = colorGradingEnabled();
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.finalizeWithACES = finalizeWithACES();
    pushArgs.useAgX = useAgX();
    pushArgs.useLegacyACES = RtxOptions::useLegacyACES();
    
    // AgX parameters
    pushArgs.agxGamma = agxGamma();
    pushArgs.agxSaturation = agxSaturation();
    pushArgs.agxExposureOffset = agxExposureOffset();
    pushArgs.agxLook = agxLook();
    pushArgs.agxContrast = agxContrast();
    pushArgs.agxSlope = agxSlope();
    pushArgs.agxPower = agxPower();

    // Tonemap args
    pushArgs.performSRGBConversion = performSRGBConversion;
    pushArgs.shadowContrast = shadowContrast();
    pushArgs.shadowContrastEnd = shadowContrastEnd();
    pushArgs.exposureFactor = exp2f(exposureBias() + RtxOptions::calcUserEVBias()); // ev100
    pushArgs.toneCurveMinStops = toneCurveMinStops();
    pushArgs.toneCurveMaxStops = toneCurveMaxStops();
    pushArgs.debugMode = tuningMode();

    // Color grad args
    pushArgs.colorBalance = colorBalance();
    pushArgs.contrast = contrast();
    pushArgs.saturation = saturation();

    // Dither args
    switch (ditherMode()) {
    case DitherMode::None: pushArgs.ditherMode = ditherModeNone; break;
    case DitherMode::Spatial: pushArgs.ditherMode = ditherModeSpatialOnly; break;
    case DitherMode::SpatialTemporal: pushArgs.ditherMode = ditherModeSpatialTemporal; break;
    }
    pushArgs.frameIndex = ctx->getDevice()->getCurrentFrameId();

    ctx->bindResourceView(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT, ctx->getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT, inputBuffer.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT, m_toneCurve.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceSampler(TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT, linearSampler);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT, colorBuffer.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ApplyTonemappingShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

  }

  void DxvkToneMapping::dispatchHDRProcessing(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::Resource& inputColorBuffer,
    const Resources::Resource& outputColorBuffer,
    const float frameTimeMilliseconds,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "HDR Processing");

    const VkExtent3D workgroups = util::computeBlockCount(inputColorBuffer.view->imageInfo().extent, VkExtent3D{ 16 , 16, 1 });

    // Prepare shader arguments
    HDRProcessingArgs pushArgs = {};
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.hdrMaxLuminance = hdrMaxLuminance();
    pushArgs.hdrMinLuminance = hdrMinLuminance();
    pushArgs.hdrPaperWhiteLuminance = hdrPaperWhiteLuminance();
    pushArgs.exposureFactor = 1.0f; // Neutral exposure factor, rely on hdrExposureBias instead
    pushArgs.frameIndex = ctx->getDevice()->getCurrentFrameId();
    pushArgs.hdrFormat = static_cast<uint32_t>(hdrFormat()); // 0=Linear, 1=PQ, 2=HLG
    pushArgs.hdrExposureBias = hdrExposureBias();
    pushArgs.hdrBrightness = hdrBrightness();
    pushArgs.hdrToneMapper = static_cast<uint32_t>(hdrToneMapper());
    pushArgs.hdrEnableDithering = hdrEnableDithering();
    pushArgs.hdrShadows = hdrShadows();
    pushArgs.hdrMidtones = hdrMidtones();
    pushArgs.hdrHighlights = hdrHighlights();
    pushArgs.hdrBlueNoiseAmplitude = hdrBlueNoiseAmplitude();

    ctx->bindResourceView(HDR_PROCESSING_BLUE_NOISE_TEXTURE, ctx->getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(HDR_PROCESSING_INPUT_BUFFER, inputColorBuffer.view, nullptr);
    ctx->bindResourceView(HDR_PROCESSING_OUTPUT_BUFFER, outputColorBuffer.view, nullptr);
    ctx->bindResourceView(HDR_PROCESSING_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceSampler(HDR_PROCESSING_EXPOSURE_INPUT, linearSampler);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, HDRProcessingShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatch(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds,
    bool performSRGBConversion,
    bool resetHistory,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Tone Mapping");

    m_resetState |= resetHistory;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    // TODO : set reset on significant camera changes as well
    if (m_toneHistogram.image.ptr() == nullptr) {
      createResources(ctx);
      m_resetState = true;
    }

    const Resources::Resource& inputColorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::Read);
    
    if (enableHDR()) {
      // HDR mode: Apply custom HDR processing with blue noise dithering
      dispatchHDRProcessing(ctx, linearSampler, exposureView, inputColorBuffer, rtOutput.m_finalOutput.resource(Resources::AccessType::Write), frameTimeMilliseconds, autoExposureEnabled);
    } else {
      // SDR mode: Apply traditional tonemapping
      if (tonemappingEnabled()) {
        dispatchHistogram(ctx, exposureView, inputColorBuffer, autoExposureEnabled);
        dispatchToneCurve(ctx);
      }

      dispatchApplyToneMapping(ctx, linearSampler, exposureView, inputColorBuffer, rtOutput.m_finalOutput.resource(Resources::AccessType::Write), performSRGBConversion, autoExposureEnabled);
    }

    m_resetState = false;
  }

  // Curve editor widget implementation
  bool DxvkToneMapping::showCurveEditor(const char* label, std::vector<CurvePoint>& points, ImVec2 size) {
    bool modified = false;
    
    ImGui::PushID(label);
    
    // Get the current draw list and canvas position
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    
    // Create invisible button for interaction
    ImGui::InvisibleButton("curve_canvas", size);
    
    // Calculate drawing area
    ImVec2 canvas_min = canvas_pos;
    ImVec2 canvas_max = ImVec2(canvas_pos.x + size.x, canvas_pos.y + size.y);
    
    // Add a border
    draw_list->AddRect(canvas_min, canvas_max, IM_COL32(255, 255, 255, 128));
    
    // Draw grid (like Photoshop)
    const int grid_lines = 4;
    for (int i = 1; i < grid_lines; ++i) {
      float t = (float)i / grid_lines;
      
      // Vertical grid lines
      float x = canvas_min.x + t * size.x;
      draw_list->AddLine(ImVec2(x, canvas_min.y), ImVec2(x, canvas_max.y), IM_COL32(128, 128, 128, 64));
      
      // Horizontal grid lines
      float y = canvas_min.y + t * size.y;
      draw_list->AddLine(ImVec2(canvas_min.x, y), ImVec2(canvas_max.x, y), IM_COL32(128, 128, 128, 64));
    }
    
    // Sort points by x coordinate
    std::sort(points.begin(), points.end(), [](const CurvePoint& a, const CurvePoint& b) {
      return a.x < b.x;
    });
    
    // Draw the curve using bezier segments
    if (points.size() >= 2) {
      const int curve_segments = 100;
      
      for (int i = 0; i < curve_segments; ++i) {
        float t1 = (float)i / curve_segments;
        float t2 = (float)(i + 1) / curve_segments;
        
        float y1 = evaluateCurve(points, t1);
        float y2 = evaluateCurve(points, t2);
        
        // Convert to screen coordinates
        ImVec2 p1 = ImVec2(canvas_min.x + t1 * size.x, canvas_max.y - y1 * size.y);
        ImVec2 p2 = ImVec2(canvas_min.x + t2 * size.x, canvas_max.y - y2 * size.y);
        
        draw_list->AddLine(p1, p2, IM_COL32(255, 255, 255, 255), 2.0f);
      }
    }
    
    // Handle mouse interactions
    ImVec2 mouse_pos = ImGui::GetMousePos();
    bool mouse_in_canvas = mouse_pos.x >= canvas_min.x && mouse_pos.x <= canvas_max.x &&
                          mouse_pos.y >= canvas_min.y && mouse_pos.y <= canvas_max.y;
    
    if (mouse_in_canvas) {
      // Convert mouse position to normalized coordinates
      float norm_x = (mouse_pos.x - canvas_min.x) / size.x;
      float norm_y = 1.0f - (mouse_pos.y - canvas_min.y) / size.y; // Flip Y axis
      
      // Clamp to valid range
      norm_x = std::max(0.0f, std::min(1.0f, norm_x));
      norm_y = std::max(0.0f, std::min(1.0f, norm_y));
      
      // Left click to add or drag points
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        // Find closest point within threshold
        int closest_point = -1;
        float min_dist = 0.05f; // Threshold for selection
        
        for (int i = 0; i < points.size(); ++i) {
          float dist = std::sqrt(std::pow(points[i].x - norm_x, 2) + std::pow(points[i].y - norm_y, 2));
          if (dist < min_dist) {
            min_dist = dist;
            closest_point = i;
          }
        }
        
        if (closest_point == -1) {
          // Add new point (but not on the endpoints)
          if (norm_x > 0.05f && norm_x < 0.95f) {
            points.emplace_back(norm_x, norm_y);
            modified = true;
          }
        }
      }
      
      // Right click to remove points
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        for (int i = 1; i < points.size() - 1; ++i) { // Don't remove endpoints
          float dist = std::sqrt(std::pow(points[i].x - norm_x, 2) + std::pow(points[i].y - norm_y, 2));
          if (dist < 0.05f) {
            points.erase(points.begin() + i);
            modified = true;
            break;
          }
        }
      }
    }
    
    // Handle dragging
    static int dragging_point = -1;
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && mouse_in_canvas) {
      if (dragging_point == -1) {
        // Find point to drag
        float norm_x = (mouse_pos.x - canvas_min.x) / size.x;
        float norm_y = 1.0f - (mouse_pos.y - canvas_min.y) / size.y;
        
        for (int i = 0; i < points.size(); ++i) {
          float dist = std::sqrt(std::pow(points[i].x - norm_x, 2) + std::pow(points[i].y - norm_y, 2));
          if (dist < 0.05f) {
            dragging_point = i;
            break;
          }
        }
      }
      
      if (dragging_point != -1) {
        float norm_x = (mouse_pos.x - canvas_min.x) / size.x;
        float norm_y = 1.0f - (mouse_pos.y - canvas_min.y) / size.y;
        
        // Clamp to valid range
        norm_x = std::max(0.0f, std::min(1.0f, norm_x));
        norm_y = std::max(0.0f, std::min(1.0f, norm_y));
        
        // Don't allow moving endpoints horizontally
        if (dragging_point == 0 || dragging_point == points.size() - 1) {
          points[dragging_point].y = norm_y;
        } else {
          points[dragging_point].x = norm_x;
          points[dragging_point].y = norm_y;
        }
        
        modified = true;
      }
    }
    
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      dragging_point = -1;
    }
    
    // Draw control points
    for (int i = 0; i < points.size(); ++i) {
      ImVec2 screen_pos = ImVec2(canvas_min.x + points[i].x * size.x, canvas_max.y - points[i].y * size.y);
      
      // Different colors for different types of points
      ImU32 color = IM_COL32(255, 255, 255, 255);
      if (i == 0 || i == points.size() - 1) {
        color = IM_COL32(255, 0, 0, 255); // Red for endpoints
      }
      
      draw_list->AddCircleFilled(screen_pos, 4.0f, color);
      draw_list->AddCircle(screen_pos, 4.0f, IM_COL32(0, 0, 0, 255), 0, 1.0f);
    }
    
    // Show curve value at mouse position
    if (mouse_in_canvas) {
      float norm_x = (mouse_pos.x - canvas_min.x) / size.x;
      float curve_y = evaluateCurve(points, norm_x);
      
      ImGui::SetTooltip("Input: %.3f, Output: %.3f", norm_x, curve_y);
    }
    
    ImGui::PopID();
    
    return modified;
  }

  // Evaluate curve at given x position using smooth interpolation
  float DxvkToneMapping::evaluateCurve(const std::vector<CurvePoint>& points, float x) {
    if (points.empty()) return x; // Linear fallback
    
    x = std::max(0.0f, std::min(1.0f, x));
    
    // Find the two points that bracket x
    int left = 0, right = points.size() - 1;
    
    for (int i = 0; i < points.size() - 1; ++i) {
      if (x >= points[i].x && x <= points[i + 1].x) {
        left = i;
        right = i + 1;
        break;
      }
    }
    
    // Linear interpolation between the two points
    if (left == right) {
      return points[left].y;
    }
    
    float t = (x - points[left].x) / (points[right].x - points[left].x);
    
    // Smooth interpolation (cubic hermite spline)
    float p0 = points[left].y;
    float p1 = points[right].y;
    
    // Calculate tangents for smooth curve
    float m0 = 0.0f, m1 = 0.0f;
    
    if (left > 0) {
      m0 = (points[right].y - points[left - 1].y) / (points[right].x - points[left - 1].x);
    }
    if (right < points.size() - 1) {
      m1 = (points[right + 1].y - points[left].y) / (points[right + 1].x - points[left].x);
    }
    
    // Apply smoothing factor
    m0 *= 0.5f;
    m1 *= 0.5f;
    
    // Hermite interpolation
    float t2 = t * t;
    float t3 = t2 * t;
    
    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 = t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 = t3 - t2;
    
    return h00 * p0 + h10 * m0 * (points[right].x - points[left].x) + h01 * p1 + h11 * m1 * (points[right].x - points[left].x);
  }

}
