/*
* Minimal FSR wrapper implementation (stub) to unblock build and UI wiring.
*/
#include "rtx_fsr.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx.h"
#include <cmath>

namespace dxvk {

  DxvkFSR::DxvkFSR(DxvkDevice* device)
  : CommonDeviceObject(device)
  , RtxPass(device) {
  }

  DxvkFSR::~DxvkFSR() {
    onDestroy();
  }

  bool DxvkFSR::isEnabled() const {
    // Enabled when UI selects FSR upscaler
    return RtxOptions::upscalerType() == UpscalerType::FSR;
  }

  float DxvkFSR::modeToScale(FSRMode mode) {
    switch (mode) {
      case FSRMode::UltraPerformance: return 0.3333f; // approx
      case FSRMode::Performance:      return 0.5f;
      case FSRMode::Balanced:         return 0.58f;
      case FSRMode::Quality:          return 0.6667f;
      case FSRMode::NativeAA:         return 1.0f;   // no upscaling, just AA path
      case FSRMode::Off:              return 1.0f;
      default:                        return 1.0f;
    }
  }

  void DxvkFSR::setSetting(const uint32_t displaySize[2], FSRMode mode, uint32_t outRenderSize[2]) {
    const float scale = modeToScale(mode);
    mOutputSize[0] = displaySize[0];
    mOutputSize[1] = displaySize[1];
    mInputSize[0] = std::max(1u, uint32_t(std::roundf(displaySize[0] * scale)));
    mInputSize[1] = std::max(1u, uint32_t(std::roundf(displaySize[1] * scale)));
    outRenderSize[0] = mInputSize[0];
    outRenderSize[1] = mInputSize[1];
  }

  void DxvkFSR::dispatch(Rc<RtxContext> ctx,
                         const Resources::RaytracingOutput& rtOutput,
                         bool resetHistory) {
    
    // Initialize FFX context if not already done
    if (!mContextInitialized) {
      // Create Vulkan backend descriptor
      ffx::CreateBackendVKDesc backendDesc = {};
      backendDesc.device = device()->handle();
      backendDesc.physicalDevice = device()->adapter()->handle();
      
      // Create upscale context descriptor
      ffx::CreateContextDescUpscale contextDesc = {};
      contextDesc.maxRenderSize = { mOutputSize[0], mOutputSize[1] };
      contextDesc.displaySize = { mOutputSize[0], mOutputSize[1] };
      contextDesc.flags = ffx::InitializationFlagBits::EnableAutoExposure;
      
      // Create the context
      auto result = ffx::CreateContext(mFfxContext, nullptr, contextDesc);
      if (result != ffx::ReturnCode::Ok) {
        Logger::warn("FSR: Failed to create FFX context");
        return;
      }
      
      mContextInitialized = true;
    }
    
    // Convert DXVK resources to FFX resources
    ffx::ApiResource colorResource = {};
    colorResource.resource = rtOutput.m_primaryDirectLightDenoised.image->handle();
    colorResource.description = {}; // Will be filled by FFX API
    
    ffx::ApiResource outputResource = {};
    outputResource.resource = rtOutput.m_finalOutput.image->handle();
    outputResource.description = {}; // Will be filled by FFX API
    
    // Setup upscale dispatch descriptor
    ffx::DispatchDescUpscale dispatchDesc = {};
    dispatchDesc.color = colorResource;
    dispatchDesc.output = outputResource;
    dispatchDesc.jitterOffset = { 0.0f, 0.0f }; // TODO: get from TAA jitter
    dispatchDesc.motionVectors = {}; // TODO: wire motion vectors if available
    dispatchDesc.exposure = {}; // Let auto-exposure handle this
    dispatchDesc.reactive = {}; // Optional reactive mask
    dispatchDesc.transparencyAndComposition = {}; // Optional T&C mask
    dispatchDesc.preExposure = 1.0f;
    dispatchDesc.renderSize = { mInputSize[0], mInputSize[1] };
    dispatchDesc.enableSharpening = true;
    dispatchDesc.sharpness = 0.8f;
    dispatchDesc.frameTimeDelta = 16.67f; // TODO: get real frame time
    dispatchDesc.reset = resetHistory;
    dispatchDesc.cameraNear = 0.1f; // TODO: get from camera
    dispatchDesc.cameraFar = 1000.0f; // TODO: get from camera
    dispatchDesc.cameraFovAngleVertical = 1.0f; // TODO: get from camera
    
    // Dispatch the upscale
    auto result = ffx::Dispatch(mFfxContext, dispatchDesc);
    if (result != ffx::ReturnCode::Ok) {
      Logger::warn("FSR: Failed to dispatch upscale");
    }
  }

  void DxvkFSR::onDestroy() {
    if (mContextInitialized) {
      ffx::DestroyContext(mFfxContext);
      mContextInitialized = false;
    }
  }
}
	}
}

