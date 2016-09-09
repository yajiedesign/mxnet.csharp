using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.single;

namespace mxnet.csharp.metric
{
    class Mae : EvalMetric
    {
        public Mae() : base("mas", 1)
        {

        }

        public override void Update(IList<NdArray> labels, IList<NdArray> preds)
        {
            CheckLabelShapes(labels, preds);
            for (int i = 0; i < labels.Count; i++)
            {
                var label = labels[i].AsNumerics();
                var pred = preds[i].AsNumerics();

                if (label.Shape.Ndim == 1)
                {
                    label = label.ReShape(new mxnet.numerics.nbase.Shape(label.Shape[0], 1));
                }

                this.SumMetric[0] += (float)SingleNArray.Abs((label - pred)).Mean();
                this.NumInst[0] += 1;
            }
        }
    }
}
