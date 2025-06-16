/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "rtx_types.h"
#include "rtx_lights.h"
#include "rtx_mod_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_utils.h"

namespace dxvk {
  class DxvkContext;

  /**
   * \brief USD Mod
   *
   * Handles asset replacements importing from a USD.
   */
  class UsdMod final : public Mod {
  public:
    ~UsdMod() override;

    void load(const Rc<DxvkContext>& context) override;
    void unload() override;
    bool checkForChanges(const Rc<DxvkContext>& context) override;

    // Get tracked USD files for debugging/display purposes
    std::vector<std::string> getTrackedFiles() const;
    
    // Check if all async mesh loading operations are complete
    bool areAsyncOperationsComplete() const;
    
    // Layer selection functionality
    std::vector<std::string> getAvailableLayers() const;
    std::vector<std::string> getEnabledLayers() const;
    void setEnabledLayers(const std::vector<std::string>& enabledLayers);
    bool isLayerEnabled(const std::string& layerPath) const;
    void setLayerEnabled(const std::string& layerPath, bool enabled);
    
    // Hierarchical layer information
    struct LayerInfo {
      std::string fullPath;
      std::string parentPath;
      std::string displayName;
      int depth;
    };
    std::vector<LayerInfo> getLayerHierarchy() const;

    static const ModTypeInfo& getTypeInfo();

    static void loadUsdPlugins();
    
  private:
    friend struct UsdModTypeInfo;
    explicit UsdMod(const Mod::Path& usdFilePath);

    // Using pimpl to hide USD from headers.
    class Impl;
    std::unique_ptr<Impl> m_impl;
  };

} // namespace dxvk
