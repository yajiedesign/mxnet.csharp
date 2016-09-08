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

        public override void Update(List<NDArray> labels, List<NDArray> preds)
        {
            check_label_shapes(labels, preds);

            for (int i = 0; i < labels.Count; i++)
            {
                var pred_label = preds[i].argmax_channel().As_numerics().ToInt32();
                var label = labels[i].As_numerics().ToInt32();
                //var t = (pred_label.Flat().Compare(label.Flat())).Data.ToArray();
                this.sum_metric[0] += (pred_label.Flat().Compare(label.Flat())).Sum();
                this.num_inst[0] += pred_label.Flat().data.Count();
            }
        }


    }
}