//
// Created by Joe Bowser on 2022-11-04.
//

#ifndef SUPERRESMLStats_MOBILECALLGUARD_H
#define SUPERRESMLStats_MOBILECALLGUARD_H


#include "torch/script.h"


namespace MLStats {

    std::string const PYTORCH_PATH = "/data/data/org.infil00p.pytorch_mobilenet_stats/files/pytorch/";

    struct MobileCallGuard
    {
        // AutoGrad is disabled for mobile by default.
        torch::autograd::AutoGradMode no_autograd_guard{false};
        // This needs to be on (taken from the test application)
        torch::AutoNonVariableTypeMode non_var_guard{true};
        // Disable graph optimizer to ensure list of unused ops are not changed for
        // custom mobile build.
        torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    };
}




#endif //SUPERRESMLStats_MOBILECALLGUARD_H
