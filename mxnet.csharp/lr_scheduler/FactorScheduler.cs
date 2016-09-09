using System;

namespace mxnet.csharp.lr_scheduler
{
    class FactorScheduler : LrScheduler
    {
        private readonly int _step;
        private readonly float _factor;
        private readonly float _stopFactorLr;
        private int _count;


        /// <summary>
        /// Reduce learning rate in factor
        /// 
        /// Assume the weight has been updated by n times, then the learning rate will be
        /// 
        /// base_lr * factor^(floor(n/step)
        /// )
        /// </summary>
        /// <param name="step">schedule learning rate after n updates</param>
        /// <param name="factor">the factor for reducing the learning rate</param>
        /// <param name="stopFactorLr"></param>
        public FactorScheduler(int step, float factor= 1f, float stopFactorLr= 1e-8f)
        {
            if (step < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(step),
                    "Schedule step must be greater or equal than 1 round");
            }
            if (factor >1.0)
            {
                throw new ArgumentOutOfRangeException(nameof(factor),
                    "Factor must be no more than 1 to make lr reduce");
            }
            this._step = step;
            this._factor = factor;
            this._stopFactorLr = stopFactorLr;
            this._count = 0;
        }

        public override float Call(int numUpdate)
        {
            if (numUpdate > this._count + this._step)
            {
                this._count += this._step;
                this.BaseLr *= this._factor;
                if (this.BaseLr < this._stopFactorLr)
                {
                    this.BaseLr = this._stopFactorLr;
                    Log.Info($"Update[{numUpdate}]: now learning rate arrived at {this.BaseLr:.5e}, will not " +
                             "change in the future");
                }
                else
                {
                    Log.Info($"Update[{numUpdate}]: Change learning rate to {this.BaseLr:.5e}");
                }
            }

            return this.BaseLr;
        }
    }
}