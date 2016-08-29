using System;
using System.Collections.Generic;

namespace mxnet.csharp.metric
{
    class Accuracy : EvalMetric
    {
        public Accuracy(string name, int num = 1) : base(name, num)
        {
        }

        public override void update(List<NDArray> label, List<NDArray> pred)
        {
      
        }
    }
}