using System;

namespace mxnet.csharp.metric
{
    class Accuracy : EvalMetric
    {
        public Accuracy(string name, int num = 1) : base(name, num)
        {
        }

        public override void update(object label, object pred)
        {
            throw new NotImplementedException();
        }
    }
}