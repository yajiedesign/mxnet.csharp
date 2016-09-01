using System;

namespace mxnet.csharp.optimizer
{
    public class Sgd : Optimizer
    {
        public override NDArray create_state(int index, NDArray weight)
        {
            throw new NotImplementedException();
        }

        public override void update(int index, NDArray weight, NDArray grad, NDArray state)
        {
            throw new NotImplementedException();
        }
    }
}