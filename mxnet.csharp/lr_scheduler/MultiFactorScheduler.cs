using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.lr_scheduler
{
    public class MultiFactorScheduler: LRScheduler
    {
        private readonly List<int> _step;
        private int _cur_step_ind;
        private readonly float _factor;
        private int _count;

        public MultiFactorScheduler(List<int> step , float factor = 1f)
        {
            for (int i = 0; i < step.Count; i++)
            {
                if (i != 0 && step[i] < step[i - 1])
                {
                    throw new ArgumentOutOfRangeException(nameof(step),
                        "Schedule step must be an increasing integer list");
                }
                if (step[i] < 1)
                {
                    throw new ArgumentOutOfRangeException(nameof(step),
                        "Schedule step must be greater or equal than 1 round");
                    
                }
            }

            if (factor > 1.0)
            {
                throw new ArgumentOutOfRangeException(nameof(step),
                    "Factor must be no more than 1 to make lr reduce");
            }
            this._step = step;
            this._cur_step_ind = 0;
            this._factor = factor;
            this._count = 0;
        }

        public override float Call(int num_update)
        {
            if (_cur_step_ind < _step.Count - 1)
            {
                if (num_update > this._step[this._cur_step_ind])
                {
                    this._count = this._step[this._cur_step_ind];
                    this._cur_step_ind += 1;
                    this.base_lr *= this._factor;
                    log.Info($"Update[{num_update}]: Change learning rate to {base_lr:.5e}");
                }
            }
            return this.base_lr;
        }
    }
}
