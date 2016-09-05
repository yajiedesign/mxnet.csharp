using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.metric
{
    class CrossEntropy: EvalMetric
    {
        public CrossEntropy() : base("cross-entropy", 1)
        {

        }

        public override void Update(List<NDArray> labels, List<NDArray> preds)
        {
            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].As_numerics();
                var pred = preds[i].As_numerics();


            }
        }
    }
}
