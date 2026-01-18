
from . import rk_sampler_beta
from . import samplers
from . import samplers_extensions


def add_beta(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers):
    # Lazy import to avoid circular dependencies
    from .rk_coefficients_beta import RK_SAMPLER_NAMES_BETA_NO_FOLDERS
    
    NODE_CLASS_MAPPINGS.update({
        #"SharkSampler"                    : samplers.SharkSampler,
        #"SharkSamplerAdvanced_Beta"       : samplers.SharkSampler, #SharkSamplerAdvanced_Beta,
        "SharkOptions_Beta"               : samplers_extensions.SharkOptions_Beta,
        "ClownOptions_SDE_Beta"           : samplers_extensions.ClownOptions_SDE_Beta,
        "ClownOptions_DetailBoost_Beta"   : samplers_extensions.ClownOptions_DetailBoost_Beta,
        "ClownGuide_Style_Beta"           : samplers_extensions.ClownGuide_Style_Beta,
        "ClownGuide_Style_EdgeWidth"      : samplers_extensions.ClownGuide_Style_EdgeWidth,
        "ClownGuide_Style_TileSize"       : samplers_extensions.ClownGuide_Style_TileSize,

        "ClownGuide_Beta"                 : samplers_extensions.ClownGuide_Beta,
        "ClownGuides_Beta"                : samplers_extensions.ClownGuides_Beta,
        "ClownGuidesAB_Beta"              : samplers_extensions.ClownGuidesAB_Beta,
        
        "ClownGuides_Sync"                : samplers_extensions.ClownGuides_Sync,
        "ClownGuides_Sync_Advanced"       : samplers_extensions.ClownGuides_Sync_Advanced,
        "ClownGuide_FrequencySeparation"  : samplers_extensions.ClownGuide_FrequencySeparation,

        
        "SharkOptions_GuiderInput"        : samplers_extensions.SharkOptions_GuiderInput,
        "ClownOptions_ImplicitSteps_Beta" : samplers_extensions.ClownOptions_ImplicitSteps_Beta,
        "ClownOptions_Cycles_Beta"        : samplers_extensions.ClownOptions_Cycles_Beta,

        "SharkOptions_GuideCond_Beta"     : samplers_extensions.SharkOptions_GuideCond_Beta,
        "SharkOptions_GuideConds_Beta"    : samplers_extensions.SharkOptions_GuideConds_Beta,
        
        "ClownOptions_Tile_Beta"          : samplers_extensions.ClownOptions_Tile_Beta,
        "ClownOptions_Tile_Advanced_Beta" : samplers_extensions.ClownOptions_Tile_Advanced_Beta,


        "ClownGuide_Mean_Beta"            : samplers_extensions.ClownGuide_Mean_Beta,
        "ClownGuide_AdaIN_MMDiT_Beta"     : samplers_extensions.ClownGuide_AdaIN_MMDiT_Beta,
        "ClownGuide_AttnInj_MMDiT_Beta"   : samplers_extensions.ClownGuide_AttnInj_MMDiT_Beta,
        "ClownGuide_StyleNorm_Advanced_HiDream" : samplers_extensions.ClownGuide_StyleNorm_Advanced_HiDream,

        "ClownOptions_SDE_Mask_Beta"      : samplers_extensions.ClownOptions_SDE_Mask_Beta,
        
        "ClownOptions_StepSize_Beta"      : samplers_extensions.ClownOptions_StepSize_Beta,
        "ClownOptions_SigmaScaling_Beta"  : samplers_extensions.ClownOptions_SigmaScaling_Beta,

        "ClownOptions_Momentum_Beta"      : samplers_extensions.ClownOptions_Momentum_Beta,
        "ClownOptions_SwapSampler_Beta"   : samplers_extensions.ClownOptions_SwapSampler_Beta,
        "ClownOptions_ExtraOptions_Beta"  : samplers_extensions.ClownOptions_ExtraOptions_Beta,
        "ClownOptions_Automation_Beta"    : samplers_extensions.ClownOptions_Automation_Beta,

        "SharkOptions_UltraCascade_Latent_Beta"  : samplers_extensions.SharkOptions_UltraCascade_Latent_Beta,
        "SharkOptions_StartStep_Beta"     : samplers_extensions.SharkOptions_StartStep_Beta,
        
        "ClownOptions_Combine"            : samplers_extensions.ClownOptions_Combine,
        "ClownOptions_Frameweights"       : samplers_extensions.ClownOptions_Frameweights,
        "ClownOptions_FlowGuide"          : samplers_extensions.ClownOptions_FlowGuide,
        
        "ClownStyle_Block_MMDiT"          : samplers_extensions.ClownStyle_Block_MMDiT,
        "ClownStyle_MMDiT"                : samplers_extensions.ClownStyle_MMDiT,
        "ClownStyle_Attn_MMDiT"           : samplers_extensions.ClownStyle_Attn_MMDiT,
        "ClownStyle_Boost"                : samplers_extensions.ClownStyle_Boost,

        "ClownStyle_UNet"                 : samplers_extensions.ClownStyle_UNet,
        "ClownStyle_Block_UNet"           : samplers_extensions.ClownStyle_Block_UNet,
        "ClownStyle_Attn_UNet"            : samplers_extensions.ClownStyle_Attn_UNet,
        "ClownStyle_ResBlock_UNet"        : samplers_extensions.ClownStyle_ResBlock_UNet,
        "ClownStyle_SpatialBlock_UNet"    : samplers_extensions.ClownStyle_SpatialBlock_UNet,
        "ClownStyle_TransformerBlock_UNet": samplers_extensions.ClownStyle_TransformerBlock_UNet,


        "ClownSamplerSelector_Beta"       : samplers_extensions.ClownSamplerSelector_Beta,

        "SharkSampler_Beta"               : samplers.SharkSampler_Beta,
        
        "SharkChainsampler_Beta"          : samplers.SharkChainsampler_Beta,

        "ClownsharKSampler_Beta"          : samplers.ClownsharKSampler_Beta,
        "ClownsharkChainsampler_Beta"     : samplers.ClownsharkChainsampler_Beta,
        
        "ClownSampler_Beta"               : samplers.ClownSampler_Beta,
        "ClownSamplerAdvanced_Beta"       : samplers.ClownSamplerAdvanced_Beta,
        
        "BongSampler"                     : samplers.BongSampler,

    })

    # Register all samplers from RK_SAMPLER_NAMES_BETA_NO_FOLDERS
    # Create sampler functions dynamically
    sampler_functions = {}
    for sampler_name in RK_SAMPLER_NAMES_BETA_NO_FOLDERS:
        if sampler_name == "none":
            continue
        
        # Create standard sampler function with proper closure
        def make_sample_fn(rk_type):
            def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type=rk_type)
            return sample_fn
        
        sampler_functions[sampler_name] = make_sample_fn(sampler_name)
        
        # Create ODE version (eta=0.0, eta_substep=0.0) for non-implicit samplers
        # Skip ODE versions for implicit samplers (those with implicit-related keywords)
        if not any(keyword in sampler_name for keyword in ["gauss-legendre", "radau", "lobatto", "irk_exp_diag", "kraaijevanger", "qin_zhang", "pareschi", "crouzeix"]):
            ode_sampler_name = f"{sampler_name}_ode"
            
            def make_sample_ode_fn(rk_type):
                def sample_ode_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type=rk_type, eta=0.0, eta_substep=0.0)
                return sample_ode_fn
            
            sampler_functions[ode_sampler_name] = make_sample_ode_fn(sampler_name)

    extra_samplers.update(sampler_functions)
    
    # Also register the generic rk_beta sampler
    extra_samplers["rk_beta"] = rk_sampler_beta.sample_rk_beta
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
            #"SharkSampler"                          : "SharkSampler",
            #"SharkSamplerAdvanced_Beta"             : "SharkSamplerAdvanced",
            "SharkSampler_Beta"                     : "SharkSampler",
            "SharkChainsampler_Beta"                : "SharkChainsampler",
            "BongSampler"                           : "BongSampler",
            "ClownsharKSampler_Beta"                : "ClownsharKSampler",
            "ClownsharkChainsampler_Beta"           : "ClownsharkChainsampler",
            "ClownSampler_Beta"                     : "ClownSampler",
            "ClownSamplerAdvanced_Beta"             : "ClownSamplerAdvanced",
            "ClownGuide_Mean_Beta"                  : "ClownGuide Mean",
            "ClownGuide_AdaIN_MMDiT_Beta"           : "ClownGuide AdaIN (HiDream)",
            "ClownGuide_AttnInj_MMDiT_Beta"         : "ClownGuide AttnInj (HiDream)",
            "ClownGuide_StyleNorm_Advanced_HiDream" : "ClownGuide_StyleNorm_Advanced_HiDream",
            "ClownGuide_Style_Beta"                 : "ClownGuide Style",
            "ClownGuide_Beta"                       : "ClownGuide",
            "ClownGuides_Beta"                      : "ClownGuides",
            "ClownGuides_Sync"                      : "ClownGuides Sync",
            "ClownGuides_Sync_Advanced"             : "ClownGuides Sync_Advanced",


            "ClownGuidesAB_Beta"                    : "ClownGuidesAB",
            "ClownSamplerSelector_Beta"             : "ClownSamplerSelector",
            "ClownOptions_SDE_Mask_Beta"            : "ClownOptions SDE Mask",
            "ClownOptions_SDE_Beta"                 : "ClownOptions SDE",
            "ClownOptions_StepSize_Beta"            : "ClownOptions Step Size",
            "ClownOptions_DetailBoost_Beta"         : "ClownOptions Detail Boost",
            "ClownOptions_SigmaScaling_Beta"        : "ClownOptions Sigma Scaling",
            "ClownOptions_Momentum_Beta"            : "ClownOptions Momentum",
            "ClownOptions_ImplicitSteps_Beta"       : "ClownOptions Implicit Steps",
            "ClownOptions_Cycles_Beta"              : "ClownOptions Cycles",
            "ClownOptions_SwapSampler_Beta"         : "ClownOptions Swap Sampler",
            "ClownOptions_ExtraOptions_Beta"        : "ClownOptions Extra Options",
            "ClownOptions_Automation_Beta"          : "ClownOptions Automation",
            "SharkOptions_GuideCond_Beta"           : "SharkOptions Guide Cond",
            "SharkOptions_GuideConds_Beta"          : "SharkOptions Guide Conds",
            "SharkOptions_Beta"                     : "SharkOptions",
            "SharkOptions_StartStep_Beta"           : "SharkOptions Start Step",
            "SharkOptions_UltraCascade_Latent_Beta" : "SharkOptions UltraCascade Latent",
            "ClownOptions_Combine"                  : "ClownOptions Combine",
            "ClownOptions_Frameweights"             : "ClownOptions Frameweights",
            "SharkOptions_GuiderInput"              : "SharkOptions Guider Input",
            "ClownOptions_Tile_Beta"                : "ClownOptions Tile",
            "ClownOptions_Tile_Advanced_Beta"       : "ClownOptions Tile Advanced",

    })
    
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers

