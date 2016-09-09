using System;

namespace mxnet.csharp.optimizer
{
    public class Sgd : Optimizer
    {
        public override NdArray create_state(int index, NdArray weight)
        {
            throw new NotImplementedException();
        }

        public override void update(int index, NdArray weight, NdArray grad, NdArray state)
        {
            throw new NotImplementedException();
        }
    }
}