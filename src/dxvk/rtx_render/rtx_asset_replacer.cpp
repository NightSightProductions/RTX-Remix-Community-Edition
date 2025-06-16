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

#include "rtx_asset_replacer.h"

#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_utils.h"
#include "rtx_asset_data_manager.h"
#include "rtx_mod_usd.h"

namespace dxvk {

std::vector<AssetReplacement>* AssetReplacer::getReplacementsForMesh(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementMeshes())
    return nullptr;

  auto variantInfo = m_variantInfos.find(hash);

  if (variantInfo != m_variantInfos.end()) {
    hash += variantInfo->second.selectedVariant;
  }

  for (auto& mod : m_modManager.mods()) {
    if (auto replacement = mod->replacements().get<AssetReplacement::eMesh>(hash)) {
      return replacement;
    }
  }

  return nullptr;
}

std::vector<AssetReplacement>* AssetReplacer::getReplacementsForLight(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementLights())
    return nullptr;

  for (auto& mod : m_modManager.mods()) {
    if (auto replacement = mod->replacements().get<AssetReplacement::eLight>(hash)) {
      return replacement;
    }
  }

  return nullptr;
}

MaterialData* AssetReplacer::getReplacementMaterial(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementMaterials())
    return nullptr;

  for (auto& mod : m_modManager.mods()) {
    MaterialData* material;
    if (mod->replacements().getObject(hash, material)) {
      return material;
    }
  }

  return nullptr;
}

void AssetReplacer::initialize(const Rc<DxvkContext>& context) {
  for (auto& mod : m_modManager.mods()) {
    mod->load(context);
  }
  updateSecretReplacements();
}

bool AssetReplacer::checkForChanges(const Rc<DxvkContext>& context) {
  ScopedCpuProfileZone();

  bool changed = false;
  for (auto& mod : m_modManager.mods()) {
    changed |= mod->checkForChanges(context);
  }
  if (changed) {
    updateSecretReplacements();
  }
  return changed;
}

bool AssetReplacer::areAllReplacementsLoaded() const {
  for (auto& mod : m_modManager.mods()) {
    if (mod->state().progressState != Mod::ProgressState::Loaded) {
      return false;
    }
  }

  return true;
}

std::vector<Mod::State> AssetReplacer::getReplacementStates() const {
  const auto& mods = m_modManager.mods();
  std::vector<Mod::State> modStates(mods.size());

  for (auto& mod : mods) {
    modStates.emplace_back(mod->state());
  }

  return modStates;
}

void AssetReplacer::updateSecretReplacements() {
  bool updated = false;

  m_variantInfos.clear();
  m_secretReplacements.clear();

  for (auto& mod : m_modManager.mods()) {
    if (mod->state().progressState != Mod::ProgressState::Loaded) {
      continue;
    }

    // Pull secret replacement info
    for (auto& secrets : mod->replacements().secretReplacements()) {
      for (auto& secret : secrets.second) {
        m_secretReplacements[secrets.first].emplace_back(
          SecretReplacement{secret.header,
                            secret.name,
                            secret.description,
                            secret.unlockHash,
                            secret.assetHash,
                            secret.replacementPath,
                            secret.bDisplayBeforeUnlocked,
                            secret.bExclusiveReplacement,
                            secret.variantId
          }
        );

        auto& numVariants = m_variantInfos[secrets.first].numVariants;
        numVariants = std::max(secret.variantId, numVariants);

        updated = true;
      }
    }
  }

  m_bSecretReplacementsUpdated = updated;
}

namespace {
  std::string tostr(const remixapi_MaterialHandle& h) {
    static_assert(sizeof h == sizeof uint64_t);
    return std::to_string(reinterpret_cast<uint64_t>(h));
  }
  std::string tostr(const remixapi_MeshHandle& h) {
    static_assert(sizeof h == sizeof uint64_t);
    return std::to_string(reinterpret_cast<uint64_t>(h));
  }
}

void AssetReplacer::makeMaterialWithTexturePreload(DxvkContext& ctx, remixapi_MaterialHandle handle, MaterialData&& data) {
  auto [iter, isNew] = m_extMaterials.emplace(handle, std::move(data));

  if (!isNew) {
    Logger::info("Ignoring repeated material registration (handle=" + tostr(handle) + ") ");
    return;
  }
}

const MaterialData* AssetReplacer::accessExternalMaterial(remixapi_MaterialHandle handle) const {
  auto found = m_extMaterials.find(handle);
  if (found == m_extMaterials.end()) {
    return nullptr;
  }
  return found->second ? &found->second.value() : nullptr;
}

void AssetReplacer::destroyExternalMaterial(remixapi_MaterialHandle handle) {
  m_extMaterials.erase(handle);
}

void AssetReplacer::registerExternalMesh(remixapi_MeshHandle handle, std::vector<RasterGeometry>&& submeshes) {
  if (m_extMeshes.count(handle) > 0) {
    Logger::info("Ignoring repeated mesh registration (handle=" + tostr(handle) + ") ");
    return;
  }

  m_extMeshes.emplace(handle, std::move(submeshes));
}

const std::vector<RasterGeometry>& AssetReplacer::accessExternalMesh(remixapi_MeshHandle handle) const {
  auto found = m_extMeshes.find(handle);
  if (found == m_extMeshes.end()) {
    static const auto s_empty = std::vector<RasterGeometry> {};
    return s_empty;
  }
  return found->second;
}

void AssetReplacer::destroyExternalMesh(remixapi_MeshHandle handle) {
  m_extMeshes.erase(handle);
}

std::vector<std::pair<std::string, std::vector<std::string>>> AssetReplacer::getTrackedUsdFiles() const {
  std::vector<std::pair<std::string, std::vector<std::string>>> result;
  
  for (const auto& mod : m_modManager.mods()) {
    std::string modPath = mod->path().string();
    std::vector<std::string> trackedFiles = mod->getTrackedFiles();
    
    if (!trackedFiles.empty()) {
      result.emplace_back(modPath, std::move(trackedFiles));
    }
  }
  
  return result;
}

std::vector<std::pair<std::string, std::vector<std::string>>> AssetReplacer::getAvailableUsdLayers() const {
  std::vector<std::pair<std::string, std::vector<std::string>>> result;
  
  for (const auto& mod : m_modManager.mods()) {
    std::string modPath = mod->path().string();
    
    // Check if this is a USD mod
    if (auto usdMod = dynamic_cast<UsdMod*>(mod.get())) {
      std::vector<std::string> availableLayers = usdMod->getAvailableLayers();
      
      if (!availableLayers.empty()) {
        result.emplace_back(modPath, std::move(availableLayers));
      }
    }
  }
  
  return result;
}

std::vector<std::pair<std::string, std::vector<std::string>>> AssetReplacer::getEnabledUsdLayers() const {
  std::vector<std::pair<std::string, std::vector<std::string>>> result;
  
  for (const auto& mod : m_modManager.mods()) {
    std::string modPath = mod->path().string();
    
    // Check if this is a USD mod
    if (auto usdMod = dynamic_cast<UsdMod*>(mod.get())) {
      std::vector<std::string> enabledLayers = usdMod->getEnabledLayers();
      
      if (!enabledLayers.empty()) {
        result.emplace_back(modPath, std::move(enabledLayers));
      }
    }
  }
  
  return result;
}

void AssetReplacer::setUsdLayerEnabled(const std::string& modPath, const std::string& layerPath, bool enabled) {
  for (const auto& mod : m_modManager.mods()) {
    if (mod->path().string() == modPath) {
      // Check if this is a USD mod
      if (auto usdMod = dynamic_cast<UsdMod*>(mod.get())) {
        usdMod->setLayerEnabled(layerPath, enabled);
        return;
      }
    }
  }
  
  Logger::warn(str::format("Could not find USD mod for path: ", modPath));
}

void AssetReplacer::refreshModsAndReloadStage(const Rc<DxvkContext>& context) {
  Logger::info("Starting full mods refresh and USD stage reload...");
  
  // Step 1: Force completion of any pending GPU operations
  Logger::info("Flushing GPU command list and waiting for device idle...");
  if (context.ptr()) {
    context->flushCommandList();
  }
  context->getDevice()->waitForIdle();
  
  // Step 2: Unload all current mods (this now properly waits for async operations)
  Logger::info("Unloading all current mods...");
  for (auto& mod : m_modManager.mods()) {
    mod->unload();
  }
  
  // Step 3: Clear all replacements
  Logger::info("Clearing all replacement data...");
  for (auto& mod : m_modManager.mods()) {
    mod->replacements().clear();
  }
  
  // Step 4: Refresh the mods directory to discover new/removed mods
  Logger::info("Refreshing mods directory...");
  m_modManager.refreshMods();
  Logger::info("Mods directory refreshed - rescanned for new/removed mods");
  
  // Step 5: Reload all mods (including any newly discovered ones)
  Logger::info("Reloading all mods...");
  for (auto& mod : m_modManager.mods()) {
    mod->load(context);
  }
  
  // Step 6: Update secret replacements
  Logger::info("Updating secret replacements...");
  updateSecretReplacements();
  
  Logger::info("Full mods refresh and USD stage reload completed successfully");
}

std::vector<std::pair<std::string, std::vector<UsdModTypes::LayerInfo>>> AssetReplacer::getUsdLayerHierarchy() const {
  std::vector<std::pair<std::string, std::vector<UsdModTypes::LayerInfo>>> result;
  
  for (const auto& mod : m_modManager.mods()) {
    std::string modPath = mod->path().string();
    
    // Check if this is a USD mod
    if (auto usdMod = dynamic_cast<UsdMod*>(mod.get())) {
      std::vector<UsdMod::LayerInfo> hierarchy = usdMod->getLayerHierarchy();
      
      if (!hierarchy.empty()) {
        std::vector<UsdModTypes::LayerInfo> convertedHierarchy;
        for (const auto& usdLayerInfo : hierarchy) {
          UsdModTypes::LayerInfo layerInfo;
          layerInfo.fullPath = usdLayerInfo.fullPath;
          layerInfo.parentPath = usdLayerInfo.parentPath;
          layerInfo.displayName = usdLayerInfo.displayName;
          layerInfo.depth = usdLayerInfo.depth;
          convertedHierarchy.push_back(layerInfo);
        }
        result.emplace_back(modPath, std::move(convertedHierarchy));
      }
    }
  }
  
  return result;
}

} // namespace dxvk

