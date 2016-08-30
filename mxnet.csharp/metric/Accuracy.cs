using System;
using System.Collections.Generic;
using System.Linq;
using mxnet.numerics.single;

namespace mxnet.csharp.metric
{
    class Accuracy : EvalMetric
    {
        public Accuracy(string name, int num) : base("accuracy")
        {
        }

        public override void update(List<NDArray> labels, List<NDArray> preds)
        {

            for (int i = 0; i < labels.Count; i++)
            {
                var pred_label = preds[i].argmax_channel().AsNumerics().ToInt32();
                var label = labels[i].AsNumerics().ToInt32();
                //var t = (pred_label.Flat().Compare(label.Flat())).Data.ToArray();
                this.sum_metric[0] += (pred_label.Flat().Compare(label.Flat())).Sum();
                this.num_inst[0] += pred_label.Flat().Data.Count();
            }
        }
    }
}