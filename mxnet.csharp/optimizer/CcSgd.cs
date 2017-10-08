using System;
using System.Collections.Generic;
using System.Globalization;
using mxnet.csharp.lr_scheduler;

namespace mxnet.csharp.optimizer
{
    public class CcSgd : Optimizer
    {
        private readonly float _momentum;

        public CcSgd(
            float momentum = 0.0f,
            float rescaleGrad = 1,
            Dictionary<int, string> paramIdx2Name = null,
            float wd = 0,
            float clipGradient = -1,
            float learningRate = 0.01F,
            LrScheduler lrScheduler = null,
            Symbol sym = null,
            int beginNumUpdate = 0)
            : base(rescaleGrad,
                paramIdx2Name,
                wd,
                clipGradient,
                learningRate,
                lrScheduler,
                sym,
                beginNumUpdate)
        {
            this._momentum = momentum;
        }

        public override NdArray create_state(int index, NdArray weight)
        {
            if (Math.Abs(this._momentum) < float.Epsilon)
            {
                return null;
            }
            return NdArray.Zeros(weight.GetShape(), weight.GetContext(), weight.GetDtype());
        }

        public override void update(int index, NdArray weight, NdArray grad, NdArray state)
        {
            var lr = this._get_lr(index);
            var wd = this._get_wd(index);
            this._update_count(index);

            if (state != null)
            {
                NdArray.SgdMomUpdate(weight, grad, state, lr, weight, this._momentum, wd, this._rescaleGrad, this._clipGradient);
            }
            else
            {
                NdArray.SgdUpdate(weight, grad, lr, weight, wd, this._rescaleGrad, this._clipGradient);
            }
        }
    }
}