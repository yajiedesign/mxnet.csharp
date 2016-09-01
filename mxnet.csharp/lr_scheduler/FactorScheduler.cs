using System;

namespace mxnet.csharp.lr_scheduler
{
    class FactorScheduler : LRScheduler
    {
        private readonly int _step;
        private readonly float _factor;
        private readonly float _stop_factor_lr;
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
        /// <param name="stop_factor_lr"></param>
        public FactorScheduler(int step, float factor= 1f, float stop_factor_lr= 1e-8f)
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
            this._stop_factor_lr = stop_factor_lr;
            this._count = 0;
        }

        public override float Call(int num_update)
        {
            if (num_update > this._count + this._step)
            {
                this._count += this._step;
                this.base_lr *= this._factor;
                if (this.base_lr < this._stop_factor_lr)
                {
                    this.base_lr = this._stop_factor_lr;
                    log.Info($"Update[{num_update}]: now learning rate arrived at {this.base_lr:.5e}, will not " +
                             "change in the future");
                }
                else
                {
                    log.Info($"Update[{num_update}]: Change learning rate to {this.base_lr:.5e}");
                }
            }

            return this.base_lr;
        }
    }
}