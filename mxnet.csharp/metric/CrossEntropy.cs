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
            check_label_shapes(labels, preds);

            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].as_numerics();
                var pred = preds[i].as_numerics();

                label = label.Flat();
                if (label.shape[0] != pred.shape[0])
                {
                    throw new ArgumentException("label and pred shape not match");
                }

                var prob = pred[Enumerable.Range(0, (int)label.shape[0]).ToArray(), label.ToInt32().data];

                this.sum_metric[0] += (-prob.Log()).Sum();
                this.num_inst[0] += (int)label.shape[0];
            }
        }
    }
}
