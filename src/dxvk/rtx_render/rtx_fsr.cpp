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
												 bool /*resetHistory*/) {
		// Stub: no operation yet. Final composite already writes to final output.
		// Later this will: setup FFX resources and call ffx::Dispatch with ffx::DispatchDescUpscale.
		(void)ctx;
		(void)rtOutput;
	}

	void DxvkFSR::onDestroy() {
		if (mContextInitialized) {
			ffx::DestroyContext(mFfxContext);
			mContextInitialized = false;
		}
	}
}

