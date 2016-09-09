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

        public override void Update(List<NdArray> labels, List<NdArray> preds)
        {
            check_label_shapes(labels, preds);

            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].AsNumerics();
                var pred = preds[i].AsNumerics();

                label = label.Flat();
                if (label.Shape[0] != pred.Shape[0])
                {
                    throw new ArgumentException("label and pred shape not match");
                }

                var prob = pred[Enumerable.Range(0, (int)label.Shape[0]).ToArray(), label.ToInt32().Data];

                this.SumMetric[0] += (-prob.Log()).Sum();
                this.NumInst[0] += (int)label.Shape[0];
            }
        }
    }
}
