using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;
using mxnet.csharp.metric;
using mxnet.csharp.util;

namespace mxnet.csharp
{
    class DataParallelExecutorManager
    {

        private ILog _logger;
        private readonly List<Tuple<int, int>> _slices;
        private readonly List<string> _argNames;
        public List<string> ParamNames { get; }
        public List<List<NdArray>> ParamArrays => _execgrp.ParamArrays;
        public List<List<NdArray>> GradArrays => _execgrp.GradArrays;

        private List<string> _auxNames;
        private readonly List<Context> _ctx
            ;

        private readonly DataParallelExecutorGroup _execgrp;
        private Symbol _symbol;
        private readonly SymbolGenerate _symGen;
        private DataParallelExecutorGroup _currExecgrp;
        private readonly Dictionary<string, DataParallelExecutorGroup> _execgrpBucket;


        public DataParallelExecutorManager(Symbol symbol, SymbolGenerate symGen, List<Context> ctx,
          IDataIter trainData, List<string> paramNames, List<string> argNames,
          List<string> auxNames, List<int> workLoadList, ILog logger)
        {
            if (logger == null)
            {
                logger = LogManager.GetLogger("");
            }
            this._logger = logger;

            var numDevice = ctx.Count;
            logger.Info("Start training with " + string.Join("", ctx));

            if (workLoadList == null)
            {
                workLoadList = Enumerable.Repeat(1, numDevice).ToList();
            }
            Util.Assert(workLoadList.Count == numDevice, "Invalid settings for work load. ");

            var slices = SplitInputSlice(trainData.BatchSize, workLoadList);

            this._slices = slices;

            this._argNames = argNames;
            this.ParamNames = paramNames;
            this._auxNames = auxNames;
            this._ctx = ctx;

            this._execgrp = new DataParallelExecutorGroup(symbol, this._argNames, this.ParamNames, this._ctx,
                this._slices, trainData);

            this._symbol = symbol;

            this._symGen = symGen;
            this._currExecgrp = null;
            // this is set when data is loaded
            if (this._symGen != null)
            {
                this._execgrpBucket = new Dictionary<string, DataParallelExecutorGroup>()
                {
                    {trainData.DefaultBucketKey,  this._execgrp }
                };
            }


        }

        private List<Tuple<int, int>> SplitInputSlice(int batchSize, List<int> workLoadList)
        {
            var totalWorkLoad = workLoadList.Sum();

            var batchNumList =
                workLoadList.Select(workLoad => Math.Round((double)workLoad * batchSize / totalWorkLoad)).ToList();
            var batchNumSum = batchNumList.Sum();

            if (batchNumSum < batchSize)
            {
                batchNumList[batchNumList.Count - 1] += batchSize - batchNumSum;
            }
            List<Tuple<int, int>> slices = new List<Tuple<int, int>>();
            int end = 0;

            foreach (var batchNum in batchNumList)
            {
                var begin = (int)Math.Min(end, batchSize);
                end = (int)Math.Min(begin + batchNum, batchSize);

                if (begin >= end)
                    throw new Exception("Too many slices such that some splits are empty");
                slices.Add(Tuple.Create(begin, end));
            }

            return slices;
        }

        public void InstallMonitor(Monitor monitor)
        {
            if (this._symGen != null)
            {
                throw new Exception("Monitoring is not implemented for bucketing");
            }
            foreach (var trainExecs in this._execgrp.TrainExecs)
            {
                monitor.Install(trainExecs);
            }
        }

        public void SetParams(Dictionary<string, NdArray> argParams, Dictionary<string, NdArray> auxParams)
        {
            foreach (var texec in _execgrp.TrainExecs)
            {
                texec.CopyParamsFrom(argParams, auxParams);
            }
     
        }

        public void LoadDataBatch(IDataBatch dataBatch)
        {
            if (this._symGen != null)
            {
                var key = dataBatch.BucketKey;
                if (!_execgrpBucket.ContainsKey(key))
                {
                    //create new bucket entry
                    var symbol = _symGen(key);
                    var execgrp = new DataParallelExecutorGroup(symbol, this._argNames,
                        this.ParamNames, this._ctx,
                        this._slices, dataBatch,
                        sharedGroup: this._execgrp);
                    this._execgrpBucket[key] = execgrp;
                }
                this._currExecgrp = this._execgrpBucket[key];
            }
            else
            {
                this._currExecgrp = this._execgrp;
            }
            this._currExecgrp.LoadDataBatch(dataBatch);
        }
        /// <summary>
        /// run forward on the current executor
        /// </summary>
        /// <param name="isTrain"></param>
        public void Forward(bool isTrain)
        {
            this._currExecgrp.Forward(isTrain: isTrain);
        }
        /// <summary>
        /// run backward on the current executor
        /// </summary>
        public void Backward()
        {
            this._currExecgrp.Backward();
        }

        public void UpdateMetric(EvalMetric metric, List<NdArray> labels)
        {
            this._currExecgrp.UpdateMetric(metric, labels);
        }
    }

    internal class DataParallelExecutorGroup
    {
        private readonly List<Dictionary<string, NdArray>> _sharedDataArrays;
        private readonly List<string> _dataNames;
        private readonly List<string> _labelNames;
        private readonly IList<string> _auxNames;
        private readonly List<int> _paramIdx;
        private readonly List<string> _paramNames;
        public List<Executor> TrainExecs { get; }
        private readonly List<List<Tuple<Tuple<int, int>, NdArray>>> _dataArrays;
        private readonly List<List<Tuple<Tuple<int, int>, NdArray>>> _labelArrays;
        public List<List<NdArray>> ParamArrays { get; }
        public List<List<NdArray>> GradArrays { get; }
        private List<List<NdArray>> _auxArrays;
        private readonly List<Tuple<int, int>> _slices;

        public DataParallelExecutorGroup(Symbol sym,
            List<string> argNames,
            List<string> paramNames,
            List<Context> ctx,
            List<Tuple<int, int>> slices,
            IDataIProvide trainData,
            DataParallelExecutorGroup sharedGroup = null)
        {
            Model.CheckArguments(sym);

            if (sharedGroup == null)
            {
                this._sharedDataArrays = ctx.Select(s => new Dictionary<string, NdArray>()).ToList();
            }
            else
            {
                this._sharedDataArrays = sharedGroup._sharedDataArrays;
            }
            this._dataNames = trainData.ProvideData.Select(s => s.Key).ToList();
            this._labelNames = trainData.ProvideLabel.Select(s => s.Key).ToList();
            this._auxNames = sym.ListAuxiliaryStates();

            this._paramIdx = argNames.Select((x, i) => new { x = x, i = i }).Where(w => paramNames.Contains(w.x)).Select(s => s.i).ToList();
            this._paramNames = _paramIdx.Select(s => argNames[s]).ToList();

            this.TrainExecs = new List<Executor>();
            for (int i = 0; i < ctx.Count; i++)
            {
                var concat = trainData.ProvideData.Concat(trainData.ProvideLabel);

                var dataShapes = concat.ToDictionary(kvK => kvK.Key,
                       kvV =>
                       {
                           List<uint> tuple = new List<uint>();
                           tuple.Add((uint)(slices[i].Item2 - slices[i].Item1));
                           tuple.AddRange(kvV.Value.Data().Skip(1));
                           return tuple.ToArray();
                       });

                var sharedExec = sharedGroup == null ? null : sharedGroup.TrainExecs[i];

                var trainExec = BindExec(sym, ctx[i], dataShapes, this._paramNames,
                    needGradInput: true, baseExec: sharedExec,
                    sharedDataArrays: this._sharedDataArrays[i]);

                this.TrainExecs.Add(trainExec);
            }


            this._dataArrays = _dataNames.Select(name => TrainExecs.Select((e, i) => Tuple.Create(slices[i], e.ArgDict[name])).ToList()).ToList();
            this._labelArrays = _labelNames.Select(name => TrainExecs.Select((e, i) => Tuple.Create(slices[i], e.ArgDict[name])).ToList()).ToList();

            this.ParamArrays = _paramIdx.Select(i => TrainExecs.Select((e) => e.ArgArrays[i]).ToList()).ToList();

            this.GradArrays = _paramIdx.Select(i => TrainExecs.Select((e) => e.GradArrays[i]).ToList()).ToList();

            this._auxArrays = Enumerable.Range(0, this._auxNames.Count).Select(i => TrainExecs.Select((e) => e.AuxArrays[i]).ToList()).ToList();

            this._slices = slices;
        }

        private  Executor BindExec(Symbol sym, Context ctx, Dictionary<string, uint[]> inputShapes,
            List<string> paramNames, bool needGradInput, Executor baseExec,
            Dictionary<string, NdArray> sharedDataArrays, Dictionary<string, Type> inputTypes = null, ILog logger = null)
        {
            if (logger == null)
            {
                logger = LogManager.GetLogger("");
            }

            var argShapes = new List<uint[]>();
            var auxShapes = new List<uint[]>();
            var outShapes = new List<uint[]>();
            sym.InferShape(inputShapes, argShapes, auxShapes, outShapes);


            var argTypes = new List<Type>();
            var auxType = new List<Type>();
            var outType = new List<Type>();
            if (inputTypes == null)
            {
                inputTypes = inputShapes.ToDictionary(k => k.Key, v => typeof(float));
            }
            sym.InferType(inputTypes, argTypes, auxType, outType);

            var gradArrays = needGradInput ? new Dictionary<string, NdArray>() : null;

            var argNames = sym.ListArguments();
            HashSet<string> needGrad;
            if (needGradInput == false)
            {
                needGrad = new HashSet<string>();
            }

            else
            {
                needGrad = new HashSet<string>(argNames.Except(inputShapes.Keys));
            }

            var gradReq = argNames.ToDictionary(name => name, v => needGrad.Contains(v) ? OpReqType.KWriteTo : OpReqType.KNullOp);

            List<NdArray> argArrays = new List<NdArray>();

            //create or borrow arguments and gradients
            for (int i = 0; i < argNames.Count; i++)
            {
                var name = argNames[i];
                if (!paramNames.Contains(name))
                {
                    NdArray argArr;
                    //data or label
                    if (sharedDataArrays != null && sharedDataArrays.ContainsKey(name))
                    {
                        argArr = sharedDataArrays[name];

                        if (Util.Prod(argArr.GetShape()) >= Util.Prod(argShapes[i]))
                        {
                            Util.Assert(argTypes[i] == argArr.GetDtype());

                            argArr = argArr.Reshape(new Shape(argShapes[i]));

                        }
                        else
                        {
                            logger.Warn($"bucketing: data \"{name}\" has a shape {new Shape(argShapes[i])}" +
                                        ", which is larger than already allocated " +
                                        $"shape {argArr.GetShape()}" +
                                        ". Need to re-allocate. Consider putting " +
                                        "default_bucket_key to be the bucket taking the largest " +
                                        "input for better memory sharing.");
                            argArr = NdArray.Zeros(new Shape(argShapes[i]), ctx, dtype: argTypes[i]);

                            // replace existing shared array because the new one is bigger
                            sharedDataArrays[name] = argArr;
                        }

                    }
                    else
                    {
                        argArr = NdArray.Zeros(new Shape(argShapes[i]), ctx, dtype: argTypes[i]);
                        if (sharedDataArrays != null)
                        {
                            sharedDataArrays[name] = argArr;
                        }

                    }

                    argArrays.Add(argArr);

                }
                else
                {
                    NdArray argArr;
                    if (baseExec == null)
                    {
                        argArr = NdArray.Zeros(new Shape(argShapes[i]), ctx, dtype: argTypes[i]);

                        if (needGradInput && needGrad.Contains(name))
                        {
                            var gradArr = NdArray.Zeros(new Shape(argShapes[i]), ctx, dtype: argTypes[i]);
                            gradArrays[name] = gradArr;
                        }

                    }
                    else
                    {
                        argArr = baseExec.ArgDict[name];
                        Util.Assert(argArr.GetShape() == new Shape(argShapes[i]));
                        Util.Assert(argArr.GetDtype() == argTypes[i]);
                        if (needGradInput && needGrad.Contains(name))
                        {
                            gradArrays[name] = baseExec.GradDict[name];
                        }
                    }
                    argArrays.Add(argArr);
                }


            }
            List<NdArray> auxArrays;
            if (baseExec == null)
            {
                auxArrays = auxShapes.Zip(auxType, (l, r) => NdArray.Zeros(new Shape(l), ctx, r)).ToList();
            }
            else
            {
                for (int i = 0; i < baseExec.AuxArrays.Count; i++)
                {
                    var a = baseExec.AuxArrays[i];
                    Util.Assert((new Shape(auxShapes[i])) == a.GetShape());
                    Util.Assert(auxType[i] == a.GetDtype());
                }
                auxArrays = baseExec.AuxArrays;
            }
     

            var executor = sym.Bind(ctx, argArrays, gradArrays,
                   gradReq, auxArrays, null, baseExec);
            return executor;
        }

        public void LoadDataBatch(IDataBatch dataBatch)
        {
            ExecutorManager.LoadData(dataBatch, this._dataArrays);
            ExecutorManager._load_label(dataBatch, this._labelArrays);
        }

        /// <summary>
        /// Perform a forward pass on each executor
        /// </summary>
        /// <param name="isTrain"></param>
        public void Forward(bool isTrain)
        {
            foreach (var texec in TrainExecs)
            {
                texec.Forward(isTrain: isTrain);
            }


        }
        /// <summary>
        /// Perform a backward pass on each executor
        /// </summary>
        public void Backward()
        {
            foreach (var texec in TrainExecs)
            {
                texec.Backward();
            }
        }
        public void UpdateMetric(EvalMetric metric, List<NdArray> labels)
        {
            for (int index = 0; index < TrainExecs.Count; index++)
            {
                var texec = TrainExecs[index];
                var islice = _slices[index];
                var labelsSlice = labels.Select(s => s.Slice((uint) islice.Item1, (uint) islice.Item2)).ToList();
                metric.Update(labelsSlice, texec.Outputs);
            }
        }
    }

    public class ExecutorManager
    {
        public static void Load_general(List<NdArray> data, List<NdArray> targets)
        {

            for (int i = 0; i < data.Count; i++)
            {
                var dSrc = data[i];
                var dTargets = targets[i];
                dSrc.CopyTo(dTargets);
            }

        }

        public static void Load_general(List<NdArray> data, List<List<Tuple<Tuple<int, int>, NdArray>>> targets)
        {

            for (int i = 0; i < data.Count; i++)
            {
                var dSrc = data[i];
                foreach (var dst in targets[i])
                {
                
                    var sliceIdx = dst.Item1;
                    var dDst = dst.Item2;
                    dSrc.Slice((uint) sliceIdx.Item1, (uint) sliceIdx.Item2).CopyTo(dDst);
                }
              
            }

        }

        public static void LoadData(IDataBatch batch, List<NdArray> targets)
        {
            Load_general(batch.Data, targets);
        }

        public static void LoadData(IDataBatch batch, List<List<Tuple<Tuple<int, int>, NdArray>>> targets)
        {
            Load_general(batch.Data, targets);
        }

        public static void _load_label(IDataBatch batch, List<NdArray> targets)
        {
            Load_general(batch.Label, targets);
        }
        public static void _load_label(IDataBatch batch, List<List<Tuple<Tuple<int, int>, NdArray>>> targets)
        {
            Load_general(batch.Label, targets);
        }
    }
}
