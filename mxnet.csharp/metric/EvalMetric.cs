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
        private readonly int num;
        private int[] num_inst;
        private float[] sum_metric;

        public EvalMetric(string name, int num = 1)
        {
            _name = name;
            this.num = num;
            reset();
        }

        public abstract void update(object label, object pred);

        void reset()
        {

            if (this.num == 1)
            {
                this.num_inst = new int[1];
                this.sum_metric = new float[1];
            }

            else
            {
                this.num_inst = new int[num];
                this.sum_metric = new float[num];
            }


        }

        public EvalMetricResult[] get_name_value()
        {
            if (num == 1)
            {
                if (this.num_inst[0] == 0)
                {
                    return new[] {new EvalMetricResult(_name, float.NaN)};
                }
                else
                {
                    return new[] {new EvalMetricResult(_name, sum_metric[0]/num_inst[0])};
                }
            }
            else
            {
                var ret = sum_metric.Zip(num_inst, (m, i) =>
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

        private static readonly Dictionary<string, Type> Metrics = new Dictionary<string, Type>
        {
            {"acc", typeof(Accuracy)},
            {"accuracy", typeof(Accuracy)},
            //{  "ce", CrossEntropy},
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
                object o = Activator.CreateInstance(Metrics[name.ToLower()], name ,1);
                return (EvalMetric) o;
            }
            catch (Exception e)
            {

                throw new Exception($"Metric must be either callable or in {string.Join(" ", Metrics.Keys) }");
            }
        }

        public static implicit operator EvalMetric(Func<object,object, float> eval)
        {
            return new CustomMetric(eval);
        }
    }
}
