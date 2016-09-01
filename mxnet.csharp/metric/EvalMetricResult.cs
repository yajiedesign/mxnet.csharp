namespace mxnet.csharp.metric
{
    public class EvalMetricResult
    {


        public EvalMetricResult(string name, float value)
        {
            this.name = name;
            this.value = value;
        }

        public string name { get;  }
        public float value { get;}
    }
}