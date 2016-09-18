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
    public delegate Symbol SymbolGenerate(string key);

    public interface IDataIProvide
    {
        Dictionary<string, Shape> ProvideData { get; }
        Dictionary<string, Shape> ProvideLabel { get; }

    }
    public interface IDataBatch : IDataIProvide
    {
        string BucketKey { get; }
        IList<NdArray> Data { get; }
        List<NdArray> Label { get; }
        int Pad { get; }
    }

    public interface IDataIter : IDataIProvide, IEnumerable<IDataBatch>
    {
        string DefaultBucketKey { get; }

        int BatchSize { get; }
        void Reset();
    }

    public partial class FeedForward
    {
        private Symbol _symbol;
        public Dictionary<string, NdArray> ArgParams { get; private set; }
        public Dictionary<string, NdArray> AuxParams { get; private set; }
        private readonly bool _allowExtraParams;
        private bool _argumentChecked;
        private readonly SymbolGenerate _symGen;
        private readonly List<Context> _ctx;
        private readonly int _numEpoch;
        private readonly Optimizer _optimizer;
        private readonly Initializer _initializer;
        private Executor _predExec;
        private readonly int _beginEpoch;
        private readonly Dictionary<string, object> _kwargs;
        private readonly int? _epochSize;

        public FeedForward(Symbol symbol = null,
            List<Context> ctx = null,
            int numEpoch = 0,
            int? epochSize = null,
            Optimizer optimizer = null,
            Initializer initializer = null,
            Dictionary<string, NdArray> argParams = null,
            Dictionary<string, NdArray> auxParams = null,
            bool allowExtraParams = false,
            int beginEpoch = 0)
            : this(symbol,
                null,
                ctx,
                numEpoch,
                epochSize,
                optimizer,
                initializer,
                argParams,
                auxParams,
                allowExtraParams,
                beginEpoch)
        {
        }

        public FeedForward(SymbolGenerate symbolGenerate = null,
            List<Context> ctx = null,
            int numEpoch = 0,
            int? epochSize = null,
            Optimizer optimizer = null,
            Initializer initializer = null,
            Dictionary<string, NdArray> argParams = null,
            Dictionary<string, NdArray> auxParams = null,
            bool allowExtraParams = false,
            int beginEpoch = 0)
            : this(null,
                symbolGenerate,
                ctx,
                numEpoch,
                epochSize,
                optimizer,
                initializer,
                argParams,
                auxParams,
                allowExtraParams,
                beginEpoch)
        {
        }

        private FeedForward(Symbol symbol = null,
            SymbolGenerate symbolGenerate = null,
            List<Context> ctx = null,
            int numEpoch = 0,
            int? epochSize = null,
            Optimizer optimizer = null,
            Initializer initializer = null,
            Dictionary<string, NdArray> argParams = null,
            Dictionary<string, NdArray> auxParams = null,
            bool allowExtraParams = false,
            int beginEpoch = 0)
        {
            this._symbol = symbol;
            this._symGen = symbolGenerate;

            if (initializer == null)
            {
                initializer = new Uniform(0.01f);
            }
            this._initializer = initializer;
            this.ArgParams = argParams;
            this.AuxParams = auxParams;
            this._allowExtraParams = allowExtraParams;

            this._argumentChecked = false;
            if (this._symGen == null)
            {
                this.CheckArguments();
            }

            if (ctx == null)
            {
                ctx = new List<mxnet.csharp.Context>() { new Context(DeviceType.KCpu, 0) };
            }

            this._ctx = ctx;
            // training parameters
            this._numEpoch = numEpoch;

            this._kwargs = new Dictionary<string, object>();

            if (optimizer == null)
            {
                optimizer = "ccsgd";
            }

            this._optimizer = optimizer;
            // internal helper state;
            this._predExec = null;
            this._beginEpoch = beginEpoch;
            this._epochSize = epochSize;

        }

        private void CheckArguments()
        {
            if (this._argumentChecked)
                return;


            Debug.Assert(this._symbol != null);
            this._argumentChecked = true;

            //check if symbol contain duplicated names.
            Model.CheckArguments(this._symbol);
            //rematch parameters to delete useless ones
            if (this._allowExtraParams)
            {
                if (this.ArgParams != null)
                {
                    var argNames = new HashSet<string>(this._symbol.ListArguments());
                    this.ArgParams = this.ArgParams.Where(w => argNames.Contains(w.Key))
                        .ToDictionary(k => k.Key, v => v.Value);
                }
                if (this.AuxParams != null)
                {
                    var auxNames = new HashSet<string>(this._symbol.ListAuxiliaryStates());
                    this.AuxParams = this.AuxParams.Where(w => auxNames.Contains(w.Key))
                        .ToDictionary(k => k.Key, v => v.Value);
                }
            }
        }

        public void Fit(IDataIter trainData,
            IDataIter evalData,
            EvalMetric evalMetric = null,
            IList<EpochEndDelegate> epochEndCallback = null,
            IList<BatchEndDelegate> batchEndCallback = null,
            string kvstoreInput = "local",
            ILog logger = null,
            IList<int> workLoadList = null, Monitor monitor = null,
            IList<BatchEndDelegate> evalBatchEndCallback = null
            )
        {



            var data = trainData;
            if (this._symGen != null)
            {
                this._symbol = this._symGen(data.DefaultBucketKey);
                this.CheckArguments();
            }
            this._kwargs["sym"] = this._symbol;

            var initParamsTemp = this.InitParams(data.ProvideData.Concat(data.ProvideLabel).ToDictionary(x => x.Key, y => y.Value));


            var argNames = initParamsTemp.Item1;
            var paramNames = initParamsTemp.Item2;
            var auxNames = initParamsTemp.Item3;

            if (evalMetric == null)
            {
                evalMetric = "acc";
            }

            //create kvstore
            var createKvstoreTemp = CreateKvstore(kvstoreInput, _ctx.Count, ArgParams);
            var kvstore = createKvstoreTemp.Item1;
            var updateOnKvstore = createKvstoreTemp.Item2;

            var paramIdx2Name = new Dictionary<int, string>();
            if (updateOnKvstore)
            {
                paramIdx2Name = paramNames.Select((x, i) => new { i = i, x = x }).ToDictionary(k => k.i, v => v.x);
            }
            else
            {
                for (int i = 0; i < paramNames.Count; i++)
                {
                    for (int k = 0; k < _ctx.Count; k++)
                    {
                        paramIdx2Name[i * _ctx.Count + k] = paramNames[i];
                    }
                }
            }
            _kwargs["param_idx2name"] = paramIdx2Name;

            //(TODO)init optmizer

            TrainMultiDevice(this._symbol, this._ctx, argNames, paramNames, auxNames,
                this.ArgParams, this.AuxParams,
                beginEpoch: this._beginEpoch, endEpoch: this._numEpoch,
                epochSize: this._epochSize,
                optimizer: _optimizer,
                trainData: data, evalData: evalData,
                evalMetric: evalMetric,
                epochEndCallback: epochEndCallback,
                batchEndCallback: batchEndCallback,
                kvstore: kvstore, updateOnKvstore: updateOnKvstore,
                logger: logger, workLoadList: workLoadList, monitor: monitor,
                evalBatchEndCallback: evalBatchEndCallback,
                symGen: this._symGen);


        }

        private static void TrainMultiDevice(Symbol symbol, 
            IList<Context> ctx, 
            IList<string> argNames,
            IList<string> paramNames,
            IList<string> auxNames,
            Dictionary<string, NdArray> argParams,
            Dictionary<string, NdArray> auxParams,
            int beginEpoch,
            int endEpoch,
            int? epochSize, 
            Optimizer optimizer,
            IDataIter trainData, 
            IDataIter evalData, 
            EvalMetric evalMetric,
            IList<EpochEndDelegate> epochEndCallback,
            IList<BatchEndDelegate> batchEndCallback,
            KvStore kvstore, bool updateOnKvstore,
            ILog logger,
            IList<int> workLoadList,
            Monitor monitor,
            IList<BatchEndDelegate> evalBatchEndCallback,
            SymbolGenerate symGen)
        {

            if (logger == null)
            {
                logger = LogManager.GetLogger("");
            }
            var executorManager = new DataParallelExecutorManager(symbol: symbol,
                 symGen: symGen,
                 ctx: ctx,
                 trainData: trainData,
                 paramNames: paramNames,
                 argNames: argNames,
                 auxNames: auxNames,
                 workLoadList: workLoadList,
                 logger: logger);


            if (monitor != null)
            {
                executorManager.InstallMonitor(monitor);

            }
            executorManager.SetParams(argParams, auxParams);

            Action<int, NdArray, NdArray> updater = null;
            if (!updateOnKvstore)
            {
                updater = Optimizer.GetUpdater(optimizer);
            }
            if (kvstore != null)
            {

                InitializeKvstore(kvstore: kvstore,
                    paramArrays: executorManager.ParamArrays,
                    argParams: argParams,
                    paramNames: executorManager.ParamNames,
                    updateOnKvstore: updateOnKvstore);
            }

            if (updateOnKvstore)
            {
                kvstore?.SetOptimizer(optimizer);
            }

            //Now start training
            for (int epoch = 0; epoch < endEpoch - beginEpoch; epoch++)
            {
                // Training phase
                Stopwatch toc = new Stopwatch();
                toc.Start();
                evalMetric.Reset();
                var nbatch = 0;
                // Iterate over training data.

                while (true)
                {
                    var doReset = true;
                    foreach (var dataBatch in trainData)
                    {
              

                        executorManager.LoadDataBatch(dataBatch);

                        monitor?.Tic();


                        executorManager.Forward(isTrain: true);
                        executorManager.Backward();

                        

                        if (updateOnKvstore)
                        {
                            UpdateParamsOnKvstore(
                                executorManager.ParamArrays,
                                executorManager.GradArrays,
                                kvstore);
                        }
                        else
                        {
                            UpdateParams(executorManager.ParamArrays,
                                executorManager.GradArrays,
                                updater: updater,
                                numDevice: ctx.Count,
                                kvstore: kvstore);
                        }
                        monitor?.TocPrint();
                        // evaluate at end, so we can lazy copy
                        executorManager.UpdateMetric(evalMetric, dataBatch.Label);

                        nbatch += 1;
                        //batch callback (for print purpose)

                        if (batchEndCallback != null)
                        {
                            var batchEndParams = new BatchEndParam(epoch: epoch,
                                 nbatch: nbatch,
                                 evalMetric: evalMetric,
                                 locals: Thread.CurrentThread.CurrentCulture);

                            foreach (var call in batchEndCallback)
                            {
                                call(batchEndParams);
                            }
                        }
                        if (epochSize != null && nbatch >= epochSize)
                        {
                            doReset = false;
                            break;
                        }


                    }

                    if (doReset)
                    {
                        logger.Info($"Epoch[{epoch}] Resetting Data Iterator");
                        trainData.Reset();
                    }

                    if (epochSize == null || nbatch >= epochSize)
                    {
                        break;
                    }

                }


                logger.Info($"Epoch[{epoch}] Time cost={(toc.ElapsedMilliseconds/1000):.000}");

                if (epochEndCallback != null || epoch + 1 == endEpoch)
                {
                    executorManager.copy_to(argParams, auxParams);
                }
          

                if (epochEndCallback != null)
                {
                    EpochEndParam epochEndParam = new EpochEndParam(epoch, symbol, argParams, auxParams);

                    foreach (var callitem in epochEndCallback)
                    {
                        callitem(epochEndParam);
                    }
                }

                // evaluation
                if (evalData!=null)
                {
                    evalMetric.Reset();
                    evalData.Reset();
                    int i = 0;
                    foreach (var eval_batch in evalData)
                    {
                        executorManager.LoadDataBatch(eval_batch);
                        executorManager.Forward(isTrain: false);
                        executorManager.UpdateMetric(evalMetric, eval_batch.Label);

                        if (evalBatchEndCallback != null)
                        {
                            var batchEndParams = new BatchEndParam(epoch: epoch,
                                 nbatch: i,
                                 evalMetric: evalMetric,
                                 locals: Thread.CurrentThread.CurrentCulture);
                            foreach (var call in evalBatchEndCallback)
                            {
                                call(batchEndParams);
                            }

                        }

                        i++;

                    }
                    var nameValue = evalMetric.get_name_value();
                    foreach (var item in nameValue)
                    {
                        logger.Info($"Epoch[{epoch}] Validation-{item.Name}={item.Value:0.000}");
                    }
                    evalData.Reset();
                }

            }
        }

        private static void UpdateParams(
            IList<List<NdArray>> paramArrays,
            IList<List<NdArray>> gradArrays,
            Action<int, NdArray, NdArray> updater,
            int numDevice, 
            KvStore kvstore = null)
        {

            for (int index = 0; index < paramArrays.Count; index++)
            {
                var argList = paramArrays[index];
                var gradList = gradArrays[index];
                if (gradList[0] == null)
                {
                    continue;
                }
                if (kvstore != null)
                {
                    //push gradient, priority is negative index
                    kvstore.Push(index, gradList, priority: -index);
                    //pull back the weights
                    kvstore.Pull(index, argList, priority: -index);
                }

                for (int k = 0; k < argList.Count; k++)
                {
                    var w = argList[k];
                    var g = gradList[k];


                    updater(index * numDevice + k, g,w);
                  
                 

                }
            }


        }

        private static void UpdateParamsOnKvstore(
            IList<List<NdArray>> paramArrays,
            IList<List<NdArray>> gradArrays,
            KvStore kvstore)
        {

            for (int index = 0; index < paramArrays.Count; index++)
            {
                var argList = paramArrays[index];
                var gradList = gradArrays[index];
                if (gradList[0] == null)
                {
                    continue;
                }
                //push gradient, priority is negative index
                kvstore.Push(index, gradList, priority: -index);
                //pull back the weights
                kvstore.Pull(index, argList, priority: -index);
            }
        }

        private static void InitializeKvstore(KvStore kvstore,
            IList<List<NdArray>> paramArrays,
            Dictionary<string, NdArray> argParams,
            IList<string> paramNames,
            bool updateOnKvstore)
        {
            for (int idx = 0; idx < paramArrays.Count; idx++)
            {
                var paramOnDevs = paramArrays[idx];
                kvstore.Init(idx, argParams[paramNames[idx]]);

                if (updateOnKvstore)
                {
                    kvstore.Pull(idx, paramOnDevs, priority: -idx);
                }

            }
        }

        private static Tuple<KvStore, bool> CreateKvstore(
            string kvstore, int count, Dictionary<string, NdArray> argParams)
        {
            KvStore kv;
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
                        var maxSize = argParams.Select(s => Util.Prod(s.Value.GetShape())).Max();
                        if (maxSize < 1024 * 1024 * 16)
                        {
                            kvstore = "local_update_cpu";
                        }
                        else
                        {
                            kvstore = "local_allreduce_cpu";
                        }
                    }
                    kv = new KvStore(kvstore);
                }

            
            }

            bool updateOnKvstore = !(kv == null || kv.Type.Contains("local_allreduce"));


            return Tuple.Create(kv, updateOnKvstore);
        }


        private Tuple<List<string>, List<string>, List<string>> InitParams(
            Dictionary<string, Shape> inputShapes,
            bool overwrite = false)
        {
            List<uint[]> argShapes = new List<uint[]>();
            List<uint[]> auxShapes = new List<uint[]>();

            this._symbol.InferShape(inputShapes, argShapes, null, auxShapes);

            var argNames = this._symbol.ListArguments();
            var inputNames = inputShapes.Keys;

            var paramNames = argNames.Except(inputNames);

            var auxNames = this._symbol.ListAuxiliaryStates();

            var paramNameShapes = argNames.Zip(argShapes, Tuple.Create).Where(w => paramNames.Contains(w.Item1));

            var argParams = paramNameShapes.ToDictionary(k => k.Item1, s => NdArray.Zeros(new Shape(s.Item2)));
            var auxParams = auxNames.Zip(auxShapes, Tuple.Create).ToDictionary(k => k.Item1, s => NdArray.Zeros(new Shape(s.Item2)));

            foreach (var kv in argParams)
            {
                var k = kv.Key;
                if (this.ArgParams != null && this.ArgParams.ContainsKey(kv.Key) && !overwrite)
                {
                    this.ArgParams[k].CopyTo(argParams[k]);
                }
                else
                {
                    this._initializer.Call(k, argParams[k]);
                }
            }


            foreach (var kv in auxParams)
            {
                var k = kv.Key;
                if (this.AuxParams != null && this.AuxParams.ContainsKey(kv.Key) && !overwrite)
                {
                    this.AuxParams[k].CopyTo(auxParams[k]);
                }
                else
                {
                    this._initializer.Call(k, argParams[k]);
                }
            }

            this.ArgParams = argParams;
            this.AuxParams = auxParams;

            return Tuple.Create(argNames.ToList(), paramNames.ToList(), auxNames.ToList());
        }


        /// <summary>
        /// Checkpoint the model checkpoint into file.
        /// The advantage of load/save is the file is language agnostic.
        /// This means the file saved using save can be loaded by other language binding of mxnet.
        /// </summary>
        /// <param name="prefix">Prefix of model name.</param>
        /// <param name="epoch"></param>
        public void Save(string prefix, int? epoch = null)
        {

            if (!epoch.HasValue)
            {
                epoch = this._numEpoch;
            }
            Model.SaveCheckpoint(prefix, epoch, this._symbol, this.ArgParams, this.AuxParams);
        }

        public static FeedForward Load(string prefix, int? epoch = null, Context ctx = null
            , int numEpoch = 0,
            int? epochSize = null,
            Optimizer optimizer = null,
            Initializer initializer = null,
            bool allowExtraParams = false,
            int beginEpoch = 0
            )
        {

            if (ctx == null)
            {
                ctx = Context.DefaultCtx;
            }

            Symbol symbol;
            Dictionary<string, NdArray> argParams;
            Dictionary<string, NdArray> auxParams;
            Model.LoadCheckpoint(prefix, epoch, out symbol, out argParams, out auxParams);

            return new FeedForward(symbol, new List<Context>() {ctx}, numEpoch, epochSize, optimizer, initializer,
                argParams, auxParams, allowExtraParams, beginEpoch);
        }
    }

}

