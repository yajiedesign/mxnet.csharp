using System;
using System.Collections.Generic;
using System.Globalization;
using mxnet.csharp.lr_scheduler;

namespace mxnet.csharp.optimizer
{
    public class CcSgd : Optimizer
    {
        private float _momentum;
        private readonly IntPtr _handle;


        public CcSgd(float momentum = 0.0f,float rescale_grad = 1, Dictionary<int, string> param_idx2_name = null, float wd = 0,
            float clip_gradient = -1, float learning_rate = 0.01F, LRScheduler lr_scheduler = null,
            Symbol sym = null, int begin_num_update = 0)
            : base(rescale_grad, param_idx2_name, wd, clip_gradient, learning_rate, lr_scheduler, sym, begin_num_update)
        {
            this._momentum = momentum;

            this._handle = Optimizer._init_cc_optimizer(
                "ccsgd",
                new[]
                {
                    "momentum",
                    "rescale_grad",
                    "clip_gradient"
                },
                new[]
                {
                    momentum.ToString(CultureInfo.InvariantCulture),
                    rescale_grad.ToString(CultureInfo.InvariantCulture),
                    clip_gradient.ToString(CultureInfo.InvariantCulture)
                });
        }

        public override NDArray create_state(int index, NDArray weight)
        {
            return null;
        }

        public override void update(int index, NDArray weight, NDArray grad, NDArray state)
        {
            var lr = this._get_lr(index);
            var wd = this._get_wd(index);
            this._update_count(index);
            Util.CallCheck(NativeMethods.MXOptimizerUpdate(this._handle,
                index,
                weight.Get_handle(),
                grad.Get_handle(),
                lr,
                wd));
        }
    }
}