using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.metric
{
    public abstract class EvalMetric
    {
        private readonly string _name;
        private readonly int _num;
        protected int[] NumInst;
        protected float[] SumMetric;

        public EvalMetric(string name, int num = 1)
        {
            this._name = name;
            this._num = num;
            Reset();
        }

        public abstract void Update(List<NdArray> labels, List<NdArray> preds);

        public void Reset()
        {

            if (this._num == 1)
            {
                this.NumInst = new int[1];
                this.SumMetric = new float[1];
            }

            else
            {
                this.NumInst = new int[_num];
                this.SumMetric = new float[_num];
            }


        }

        public EvalMetricResult[] get_name_value()
        {
            if (_num == 1)
            {
                if (this.NumInst[0] == 0)
                {
                    return new[] {new EvalMetricResult(_name, float.NaN)};
                }
                else
                {
                    return new[] {new EvalMetricResult(_name, SumMetric[0]/NumInst[0])};
                }
            }
            else
            {
                var ret = SumMetric.Zip(NumInst, (m, i) =>
                {
                    if (i == 0)
                    {
                        return new EvalMetricResult(_name, float.NaN);
                    }
                    else
                    {
                        return new EvalMetricResult(_name, m/i);
                    }

                });
                return ret.ToArray();
            }
        }

        protected static void check_label_shapes(List<NdArray> labels, List<NdArray> preds)
        {
            if (labels.Count != preds.Count)
            {
                throw new ArgumentException($"Shape of labels does not match shape of predictions {labels.Count} {preds.Count}"  );
            }
        }

        private static readonly Dictionary<string, Type> Metrics = new Dictionary<string, Type>
        {
            {"acc", typeof(Accuracy)},
            {"accuracy", typeof(Accuracy)},
            {  "ce", typeof(CrossEntropy)},
            //{  "f1", F1},
            //{  "mae", MAE},
            //{  "mse", MSE},
            //{  "rmse", RMSE},
            //{  "top_k_accuracy", TopKAccuracy}
        };

        public static implicit operator EvalMetric(string name)
        {

            try
            {
                object o = Activator.CreateInstance(Metrics[name.ToLower()]);
                return (EvalMetric) o;
            }
            catch (Exception)
            {
                throw new Exception($"Metric must be either callable or in {string.Join(" ", Metrics.Keys) }");
            }
        }

        public static implicit operator EvalMetric(CustomMetricEval eval)
        {
            return new CustomMetric(eval);
        }
    }
}
