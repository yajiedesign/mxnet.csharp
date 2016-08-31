using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using log4net;
using mxnet.csharp.initializer;
using mxnet.csharp.metric;
using mxnet.csharp.optimizer;
using mxnet.csharp.util;

namespace mxnet.csharp
{
    public interface IDataIProvide
    {
        Dictionary<string, Shape> provide_data { get; }
        Dictionary<string, Shape> provide_label { get; }

    }
    public interface IDataBatch : IDataIProvide
    {
        string bucket_key { get; }
        List<NDArray> Data { get; }
        List<NDArray> Label { get; }
    }

    public interface IDataIter : IDataIProvide, IEnumerable<IDataBatch>
    {
        string default_bucket_key { get; }

        int batch_size { get; }
        void reset();
    }

    public class FeedForward
    {
        private Symbol symbol;
        private Dictionary<string, NDArray> arg_params;
        private Dictionary<string, NDArray> aux_params;
        private bool allow_extra_params;
        private bool argument_checked;
        private Func<string, Symbol> sym_gen;
        private List<Context> ctx;
        private int num_epoch;
        private Optimizer optimizer;
        private Initializer initializer;
        private object _pred_exec;
        private int begin_epoch;
        private Dictionary<string, object> kwargs;
        private int? epoch_size;


        public FeedForward(Symbol symbol,
            List<Context> ctx = null,
            int num_epoch = 0,
            int? epoch_size =null,
            Optimizer optimizer = null,
            Initializer initializer = null,
            Dictionary<string, NDArray> arg_params = null,
            Dictionary<string, NDArray> aux_params = null,
            bool allow_extra_params = false,
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
                optimizer = "ccsgd";
            }

            this.optimizer = optimizer;
            // internal helper state;
            this._pred_exec = null;
            this.begin_epoch = begin_epoch;
            this.epoch_size = epoch_size;

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
            List<Action> epoch_end_callback = null,
             List<Action<BatchEndParam>> batch_end_callback = null,
            string kvstore_input = "local",
            ILog logger = null,
            List<int> work_load_list = null, Monitor monitor = null,
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

            var param_idx2name = new Dictionary<int, string>();
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
                        param_idx2name[i * ctx.Count + k] = param_names[i];
                    }
                }
            }
            kwargs["param_idx2name"] = param_idx2name;

            //(TODO)init optmizer

            _train_multi_device(this.symbol, this.ctx, arg_names, param_names, aux_names,
                this.arg_params, this.aux_params,
                begin_epoch: this.begin_epoch, end_epoch: this.num_epoch,
                epoch_size: this.epoch_size,
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

        private static void _train_multi_device(Symbol symbol, List<Context> ctx, List<string> arg_names,
            List<string> param_names, List<string> aux_names, Dictionary<string, NDArray> arg_params,
            Dictionary<string, NDArray> aux_params, int begin_epoch, int end_epoch,int? epoch_size, Optimizer optimizer,
            IDataIter train_data, IDataIter eval_data, EvalMetric eval_metric, List<Action> epoch_end_callback,
            List<Action<BatchEndParam>> batch_end_callback, KVStore kvstore, bool update_on_kvstore, ILog logger, List<int> work_load_list,
            Monitor monitor, Action eval_batch_end_callback, Func<string, Symbol> sym_gen)
        {

            if (logger == null)
            {
                logger = LogManager.GetLogger("");
            }
            var executor_manager = new DataParallelExecutorManager(symbol: symbol,
                 sym_gen: sym_gen,
                 ctx: ctx,
                 train_data: train_data,
                 param_names: param_names,
                 arg_names: arg_names,
                 aux_names: aux_names,
                 work_load_list: work_load_list,
                 logger: logger);


            if (monitor != null)
            {
                executor_manager.install_monitor(monitor);

            }

   

            executor_manager.set_params(arg_params, aux_params);

            var temp2 = executor_manager.param_arrays.First().First().AsNumerics();
            var temp1 = executor_manager.grad_arrays.First().First().AsNumerics();

            Action<int, NDArray, NDArray> updater = null;
            if (!update_on_kvstore)
            {
                updater = Optimizer.get_updater(optimizer);
            }
            if (kvstore != null)
            {

                _initialize_kvstore(kvstore: kvstore,
                    param_arrays: executor_manager.param_arrays,
                    arg_params: arg_params,
                    param_names: executor_manager.param_names,
                    update_on_kvstore: update_on_kvstore);
            }

            if (update_on_kvstore)
            {
                kvstore.set_optimizer(optimizer);
            }

            //Now start training
            for (int epoch = 0; epoch < end_epoch - begin_epoch; epoch++)
            {
                // Training phase
                Stopwatch toc = new Stopwatch();
                toc.Start();
                eval_metric.reset();
                var nbatch = 0;
                // Iterate over training data.

                while (true)
                {
                    var do_reset = true;
                    foreach (var data_batch in train_data)
                    {
              

                        executor_manager.load_data_batch(data_batch);

                        monitor?.tic();


                        executor_manager.forward(is_train: true);
                        executor_manager.backward();

                        

                        if (update_on_kvstore)
                        {
                            _update_params_on_kvstore(
                                executor_manager.param_arrays,
                                executor_manager.grad_arrays,
                                kvstore);
                        }
                        else
                        {
                            _update_params(executor_manager.param_arrays,
                                executor_manager.grad_arrays,
                                updater: updater,
                                num_device: ctx.Count,
                                kvstore: kvstore);
                        }
                        monitor?.toc_print();
                        // evaluate at end, so we can lazy copy
                        executor_manager.update_metric(eval_metric, data_batch.Label);

                        nbatch += 1;
                        //batch callback (for print purpose)

                        if (batch_end_callback != null)
                        {
                            var batch_end_params = new BatchEndParam(epoch: epoch,
                                 nbatch: nbatch,
                                 eval_metric: eval_metric,
                                 locals: Thread.CurrentThread.CurrentCulture);

                            foreach (var call in batch_end_callback)
                            {
                                call(batch_end_params);
                            }
                        }
                        if (epoch_size != null && nbatch >= epoch_size)
                        {
                            do_reset = false;
                            break;
                        }


                    }

                    if (do_reset)
                    {
                        logger.Info($"Epoch[{epoch}] Resetting Data Iterator");
                        train_data.reset();
                    }

                    if (epoch_size == null || nbatch >= epoch_size)
                    {
                        break;
                    }

                }


                logger.Info($"Epoch[{epoch}] Time cost={(toc.ElapsedMilliseconds/1000):.000}");

            }
        }

        private static void _update_params(
            List<List<NDArray>> param_arrays,
            List<List<NDArray>> grad_arrays,
            Action<int, NDArray, NDArray> updater,
            int num_device, KVStore kvstore = null)
        {

            for (int index = 0; index < param_arrays.Count; index++)
            {
                var arg_list = param_arrays[index];
                var grad_list = grad_arrays[index];
                if (grad_list[0] == null)
                {
                    continue;
                }
                if (kvstore != null)
                {
                    //push gradient, priority is negative index
                    kvstore.Push(index, grad_list, priority: -index);
                    //pull back the weights
                    kvstore.Pull(index, arg_list, priority: -index);
                }

                for (int k = 0; k < arg_list.Count; k++)
                {
                    var w = arg_list[k];
                    var g = grad_list[k];


                    updater(index * num_device + k, w,g);
                  
                 

                }
            }


        }

        private static void _update_params_on_kvstore(
            List<List<NDArray>> param_arrays,
            List<List<NDArray>> grad_arrays,
            KVStore kvstore)
        {

            for (int index = 0; index < param_arrays.Count; index++)
            {
                var arg_list = param_arrays[index];
                var grad_list = grad_arrays[index];
                if (grad_list[0] == null)
                {
                    continue;
                }
                //push gradient, priority is negative index
                kvstore.Push(index, grad_list, priority: -index);
                //pull back the weights
                kvstore.Pull(index, arg_list, priority: -index);
            }
        }

        private static void _initialize_kvstore(KVStore kvstore,
            List<List<NDArray>> param_arrays,
            Dictionary<string, NDArray> arg_params,
            List<string> param_names,
            bool update_on_kvstore)
        {
            for (int idx = 0; idx < param_arrays.Count; idx++)
            {
                var param_on_devs = param_arrays[idx];
                kvstore.Init(idx, arg_params[param_names[idx]]);

                if (update_on_kvstore)
                {
                    kvstore.Pull(idx, param_on_devs, priority: -idx);
                }

            }
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
                        var max_size = argParams.Select(s => Util.Prod(s.Value.GetShape())).Max();
                        if (max_size < 1024 * 1024 * 16)
                        {
                            kvstore = "local_update_cpu";
                        }
                        else
                        {
                            kvstore = "local_allreduce_cpu";
                        }
                    }
                    kv = new KVStore(kvstore);
                }

            
            }

            bool update_on_kvstore = !(kv == null || kv.Type.Contains("local_allreduce"));


            return Tuple.Create(kv, update_on_kvstore);
        }


        private Tuple<List<string>, List<string>, List<string>> _init_params(Dictionary<string, Shape> input_shapes, bool overwrite = false)
        {
            List<uint[]> arg_shapes = new List<uint[]>();
            List<uint[]> aux_shapes = new List<uint[]>(); ;

            this.symbol.InferShape(input_shapes, arg_shapes, null, aux_shapes);

            var arg_names = this.symbol.ListArguments();
            var input_names = input_shapes.Keys;

            var param_names = arg_names.Except(input_names);

            var aux_names = this.symbol.ListAuxiliaryStates();

            var param_name_shapes = arg_names.Zip(arg_shapes, Tuple.Create).Where(w => param_names.Contains(w.Item1));

            var arg_params = param_name_shapes.ToDictionary(k => k.Item1, s => NDArray.Zeros(new Shape(s.Item2)));
            var aux_params = aux_names.Zip(aux_shapes, Tuple.Create).ToDictionary(k => k.Item1, s => NDArray.Zeros(new Shape(s.Item2)));

            foreach (var kv in arg_params)
            {
                var k = kv.Key;
                if (this.arg_params != null && this.arg_params.ContainsKey(kv.Key) && !overwrite)
                {
                    this.arg_params[k].CopyTo(arg_params[k]);
                }
                else
                {
                    this.initializer.Call(k, arg_params[k]);
                }
            }


            foreach (var kv in aux_params)
            {
                var k = kv.Key;
                if (this.aux_params != null && this.aux_params.ContainsKey(kv.Key) && !overwrite)
                {
                    this.aux_params[k].CopyTo(aux_params[k]);
                }
                else
                {
                    this.initializer.Call(k, arg_params[k]);
                }
            }

            this.arg_params = arg_params;
            this.aux_params = aux_params;

            return Tuple.Create(arg_names.ToList(), param_names.ToList(), aux_names.ToList());
        }
    }
}
