using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.csharp.initializer;

namespace mxnet.csharp
{
    public interface IDataIter
    {
        string default_bucket_key { get; set; }
        Dictionary<string,Shape> provide_data { get; set; }
        Dictionary<string, Shape> provide_label { get; set; }
    }

    public class FeedForward
    {
        private Symbol symbol;
        private Dictionary<string, NDArray> arg_params;
        private Dictionary<string, NDArray> aux_params;
        private bool allow_extra_params;
        private bool argument_checked;
        private Func<string,Symbol> sym_gen;
        private List<Context> ctx;
        private int num_epoch;
        private string optimizer;
        private Initializer initializer;
        private object _pred_exec;
        private int begin_epoch;
        private Dictionary<string,object> kwargs;


        public FeedForward(Symbol symbol, 
            List<Context> ctx = null,
            int num_epoch = 0, 
            string optimizer = "sgd",
            Initializer initializer =  null,
            Dictionary<string, NDArray> arg_params = null,
            Dictionary<string, NDArray> aux_params = null,
            bool allow_extra_params =false,
            int begin_epoch = 0)
        {
            this.symbol = symbol;

            if (initializer == null)
            {
                initializer = new Uniform(0.01f);
            }
            this.initializer = initializer;
            this.arg_params = arg_params;
            this.aux_params = aux_params;
            this.allow_extra_params = allow_extra_params;

            this.argument_checked = false;
            if (this.sym_gen == null)
            {
                this._check_arguments();
            }

            if (ctx == null)
            {
                ctx = new List<mxnet.csharp.Context>() { new Context(DeviceType.KCpu, 0) };
            }

            this.ctx = ctx;
        // training parameters
            this.num_epoch = num_epoch;

            this.kwargs = new Dictionary<string, object>();


            this.optimizer = optimizer;
        // internal helper state;
            this._pred_exec = null;
            this.begin_epoch = begin_epoch;

        }

        private void _check_arguments()
        {
            if (this.argument_checked)
                return;

            Debug.Assert(this.symbol != null);
            this.argument_checked = true;

            //check if symbol contain duplicated names.
            _check_arguments(this.symbol);
            //rematch parameters to delete useless ones
            if (this.allow_extra_params)
            {
                if (this.arg_params != null)
                {
                    var argNames = new HashSet<string>(this.symbol.ListArguments());
                    this.arg_params = this.arg_params.Where(w => argNames.Contains(w.Key))
                        .ToDictionary(k => k.Key, v => v.Value);
                }
                if (this.aux_params != null)
                {
                    var auxNames = new HashSet<string>(this.symbol.ListAuxiliaryStates());
                    this.aux_params = this.aux_params.Where(w => auxNames.Contains(w.Key))
                        .ToDictionary(k => k.Key, v => v.Value);
                }
            }
        }

        private void _check_arguments(Symbol symbol)
        {
            var argNames = symbol.ListArguments();
            var argNamesDuplicate = argNames.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));
            foreach (var name in argNamesDuplicate)
            {
                throw new Exception($"Find duplicated argument name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {string.Join(" ", argNames)}");
            }
            var auxNames = symbol.ListAuxiliaryStates();
            var auxNamesDuplicate = auxNames.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));

            foreach (var name in auxNamesDuplicate)
            {
                throw new Exception($"Find duplicated auxiliary name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {string.Join(" ", argNames)}");
            }
        }

        public void Fit(IDataIter trainData,
            IDataIter evalData,
            metric.EvalMetric eval_metric = null,
            Action epoch_end_callback = null,
            Action batch_end_callback = null,
            string kvstore = "local",
            Action logger = null,
            object work_load_list = null, object monitor = null,
            Action eval_batch_end_callback = null
            )
        {



            var data = trainData;
            if (this.sym_gen != null)
            {
                this.symbol = this.sym_gen(data.default_bucket_key);
                this._check_arguments();
            }
            this.kwargs["sym"] = this.symbol;

            var _init_params_temp = this._init_params(data.provide_data.Concat(data.provide_label).ToDictionary(x => x.Key, y => y.Value));


            var arg_names = _init_params_temp.Item1;
            var param_names = _init_params_temp.Item2;
            var aux_names = _init_params_temp.Item3;

            if (eval_metric == null)
            {
                eval_metric = "acc";
            }


        }


        private Tuple<List<string>, List<string>, List<string>> _init_params(Dictionary<string, Shape> input_shapes ,bool overwrite =false)
        {
            List<uint[]> arg_shapes = new List<uint[]>();
            List<uint[]> aux_shapes = new List<uint[]>(); ;

            this.symbol.InferShape(input_shapes, arg_shapes, null, aux_shapes);

            var arg_names = this.symbol.ListArguments();
            var input_names = input_shapes.Keys;

            var param_names = arg_names.Except(input_names);

            var aux_names = this.symbol.ListAuxiliaryStates();

            var param_name_shapes = arg_names.Zip(arg_shapes, Tuple.Create).Where(w => param_names.Contains(w.Item1));

            var arg_params = param_name_shapes.ToDictionary(k=>k.Item1, s => new NDArray(s.Item2));
            var aux_params = aux_names.Zip(aux_shapes, Tuple.Create).ToDictionary(k => k.Item1, s => new NDArray(s.Item2));

            foreach (var kv in arg_params)
            {
                var k = kv.Key;
                if (this.arg_params != null && this.arg_params.ContainsKey(kv.Key) && !overwrite)
                {
                    arg_params[k].CopyTo(arg_params[k]);
                }
            }


            foreach (var kv in aux_params)
            {
                var k = kv.Key;
                if (this.aux_params != null && this.aux_params.ContainsKey(kv.Key) && !overwrite)
                {
                    aux_params[k].CopyTo(aux_params[k]);
                }
            }

            this.arg_params = arg_params;
            this.aux_params = aux_params;

            return Tuple.Create(arg_names.ToList(), param_names.ToList(), aux_names.ToList());
        }
    }
}
