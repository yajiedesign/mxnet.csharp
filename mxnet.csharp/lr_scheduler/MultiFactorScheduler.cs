using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.lr_scheduler
{
    public class MultiFactorScheduler: LrScheduler
    {
        private readonly IList<int> _step;
        private int _curStepInd;
        private readonly float _factor;
        private int _count;

        public MultiFactorScheduler(IList<int> step , float factor = 1f)
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
            this._curStepInd = 0;
            this._factor = factor;
            this._count = 0;
        }

        public override float Call(int numUpdate)
        {
            if (_curStepInd < _step.Count - 1)
            {
                if (numUpdate > this._step[this._curStepInd])
                {
                    this._count = this._step[this._curStepInd];
                    this._curStepInd += 1;
                    this.BaseLr *= this._factor;
                    Log.Info($"Update[{numUpdate}]: Change learning rate to {BaseLr:.5e}");
                }
            }
            return this.BaseLr;
        }
    }
}
