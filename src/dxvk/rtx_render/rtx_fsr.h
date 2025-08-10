/*
* Copyright (c) 2025.
*/
#pragma once

#include "../dxvk_include.h"
#include "rtx_resources.h"
#include "dxvk_image.h"
#include "rtx_common_object.h"
#include "rtx_options.h"

// FidelityFX SDK (FFX API)
#include "../../../submodules/ffx_api/include/ffx_api/ffx_api.hpp"
#include "../../../submodules/ffx_api/include/ffx_api/ffx_upscale.hpp"
#include "../../../submodules/ffx_api/include/ffx_api/vk/ffx_api_vk.hpp"

namespace dxvk {

	// Minimal FSR wrapper; stubbed functionality for now.
	class DxvkFSR : public CommonDeviceObject, public RtxPass {
	public:
		explicit DxvkFSR(DxvkDevice* device);
		~DxvkFSR();

		// Configure based on display size and selected mode. Returns render size.
		void setSetting(const uint32_t displaySize[2], FSRMode mode, uint32_t outRenderSize[2]);

		// Getters for sizes (input = render, output = display)
		void getInputSize(uint32_t& w, uint32_t& h) const { w = mInputSize[0]; h = mInputSize[1]; }
		void getOutputSize(uint32_t& w, uint32_t& h) const { w = mOutputSize[0]; h = mOutputSize[1]; }

		// Main dispatch; no-ops for now. History reset optional.
		void dispatch(Rc<RtxContext> ctx,
									const Resources::RaytracingOutput& rtOutput,
									bool resetHistory = false);

		void onDestroy() override;

	protected:
		bool isEnabled() const override;

	private:
		// Translate FSR quality mode to scale factor.
		static float modeToScale(FSRMode mode);

		// Cached sizes
		uint32_t mInputSize[2] = { 0, 0 };
		uint32_t mOutputSize[2] = { 0, 0 };

		// FFX Context (unused until full implementation)
		ffx::Context mFfxContext = {};
		bool mContextInitialized = false;
	};
}

