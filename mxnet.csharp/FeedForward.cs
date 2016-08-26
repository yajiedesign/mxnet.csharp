using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;
using mxnet.csharp.initializer;
using mxnet.csharp.metric;

namespace mxnet.csharp
{
    public interface IDataIter
    {
        string default_bucket_key { get; set; }
        Dictionary<string,Shape> provide_data { get; set; }
        Dictionary<string, Shape> provide_label { get; set; }
        int batch_size { get; }
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
        private Optimizer optimizer;
        private Initializer initializer;
        private object _pred_exec;
        private int begin_epoch;
        private Dictionary<string,object> kwargs;


        public FeedForward(Symbol symbol, 
            List<Context> ctx = null,
            int num_epoch = 0, 
            Optimizer optimizer = null,
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

            if (optimizer == null)
            {
                float learning_rate = 1e-4f;
                float weight_decay = 1e-4f;
                optimizer = new Optimizer("ccsgd", learning_rate, weight_decay);
                optimizer.SetParam("momentum", 0.9)
                    .SetParam("rescale_grad", 1.0)
                    .SetParam("clip_gradient", 10);
            }

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
            Util._check_arguments(this.symbol);
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

        public void Fit(IDataIter trainData,
            IDataIter evalData,
            metric.EvalMetric eval_metric = null,
            Action epoch_end_callback = null,
            Action batch_end_callback = null,
            string kvstore_input = "local",
            ILog logger = null,
            List<int> work_load_list = null, object monitor = null,
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

            //create kvstore
            var _create_kvstore_temp = _create_kvstore(kvstore_input, ctx.Count, arg_params);
            var kvstore = _create_kvstore_temp.Item1;
            var update_on_kvstore = _create_kvstore_temp.Item2;

            var param_idx2name =new  Dictionary<int, string>();
            if (update_on_kvstore)
            {
                param_idx2name = param_names.Select((x, i) => new { i = i, x = x }).ToDictionary(k => k.i, v => v.x);
            }
            else
            {
                for (int i = 0; i < param_names.Count; i++)
                {
                    for (int k = 0; k < ctx.Count; k++)
                    {
                        param_idx2name[i*ctx.Count + k] = param_names[i];
                    }
                }
            }
            kwargs["param_idx2name"] = param_idx2name;

            //(TODO)init optmizer

            _train_multi_device(this.symbol, this.ctx, arg_names, param_names, aux_names,
                this.arg_params, this.aux_params,
                begin_epoch: this.begin_epoch, end_epoch: this.num_epoch,
                //epoch_size = this.epoch_size,
                optimizer: optimizer,
                train_data: data, eval_data: evalData,
                eval_metric: eval_metric,
                epoch_end_callback: epoch_end_callback,
                batch_end_callback: batch_end_callback,
                kvstore: kvstore, update_on_kvstore: update_on_kvstore,
                logger: logger, work_load_list: work_load_list, monitor: monitor,
                eval_batch_end_callback: eval_batch_end_callback,
                sym_gen: this.sym_gen);


        }

        private void _train_multi_device(Symbol symbol1, List<Context> contexts, List<string> argNames,
            List<string> paramNames, List<string> auxNames, Dictionary<string, NDArray> argParams,
            Dictionary<string, NDArray> auxParams, int begin_epoch, int end_epoch, Optimizer optimizer,
            IDataIter train_data, IDataIter eval_data, EvalMetric eval_metric, Action epoch_end_callback,
            Action batch_end_callback, KVStore kvstore, bool update_on_kvstore, ILog logger, List<int> work_load_list,
            object monitor, Action eval_batch_end_callback, Func<string, Symbol> sym_gen)
        {

            if (logger == null)
            {
                logger = LogManager.GetLogger("");
            }
           var executor_manager = new DataParallelExecutorManager(symbol: symbol,
                sym_gen: sym_gen,
                ctx: ctx,
                train_data: train_data,
                param_names: paramNames,
                arg_names: argNames,
                aux_names: auxNames,
                work_load_list: work_load_list,
                logger: logger);
        }

        private static Tuple<KVStore, bool> _create_kvstore(string kvstore, int count, Dictionary<string, NDArray> argParams)
        {
            KVStore kv = null;
            if (kvstore == null)
            {
                kv = null;
            }
            else
            {
                if (count == 1 && !kvstore.Contains("dist"))
                {
                    kv = null;
                }
                else
                {
                    if (kvstore == "local")
                    {

                        //automatically select a proper local
                       var  max_size = argParams.Select(s =>Util.Prod(s.Value.GetShape())).Max();
                        if (max_size < 1024*1024*16)
                        {
                            kvstore = "local_update_cpu";
                        }
                        else
                        {
                            kvstore = "local_allreduce_cpu";
                        }
                    }

                }

                kv = new KVStore(kvstore);
            }

            bool update_on_kvstore = !(kv==null  || kv.GetType().Contains("local_allreduce"));


            return Tuple.Create(kv, update_on_kvstore);
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
