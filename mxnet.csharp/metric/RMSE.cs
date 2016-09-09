using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.single;

namespace mxnet.csharp.metric
{
    class RMSE : EvalMetric
    {
        public RMSE() : base("rmse", 1)
        {

        }

        public override void Update(List<NDArray> labels, List<NDArray> preds)
        {
            check_label_shapes(labels, preds);
            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].as_numerics();
                var pred = preds[i].as_numerics();

                if (label.shape.ndim == 1)
                {
                    label = label.ReShape(new mxnet.numerics.nbase.Shape(label.shape[0], 1));
                }

                this.sum_metric[0] += (float)Math.Sqrt(SingleNArray.Pow((label - pred), 2).Mean());
                this.num_inst[0] += 1;
            }
        }
    }
}
