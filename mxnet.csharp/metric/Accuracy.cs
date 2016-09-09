using System;
using System.Collections.Generic;
using System.Linq;
using mxnet.numerics.single;

namespace mxnet.csharp.metric
{
    class Accuracy : EvalMetric
    {
        public Accuracy() : base("accuracy")
        {
        }

        public override void Update(List<NdArray> labels, List<NdArray> preds)
        {
            check_label_shapes(labels, preds);

            for (int i = 0; i < labels.Count; i++)
            {
                var predLabel = preds[i].ArgmaxChannel().AsNumerics().ToInt32();
                var label = labels[i].AsNumerics().ToInt32();
                //var t = (pred_label.Flat().Compare(label.Flat())).Data.ToArray();
                this.SumMetric[0] += (predLabel.Flat().Compare(label.Flat())).Sum();
                this.NumInst[0] += predLabel.Flat().Data.Count();
            }
        }


    }
}