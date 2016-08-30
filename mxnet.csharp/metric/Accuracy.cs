using System;
using System.Collections.Generic;

namespace mxnet.csharp.metric
{
    class Accuracy : EvalMetric
    {
        public Accuracy(string name, int num = 1) : base(name, num)
        {
        }

        public override void update(List<NDArray> labels, List<NDArray> preds)
        {

            for (int i = 0; i < labels.Count; i++)
            {
                var pred_label = preds[i].argmax_channel().AsNumerics().ToInt32();
                var label = labels[i].AsNumerics().ToInt32();


            }
        }
    }
}