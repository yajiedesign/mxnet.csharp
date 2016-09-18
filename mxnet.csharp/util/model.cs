using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;

namespace mxnet.csharp.util
{
    static class Model
    {
        public static void CheckArguments(Symbol symbol)
        {
            var argNames = symbol.ListArguments();
            var argNamesDuplicate = argNames.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));
            foreach (var name in argNamesDuplicate)
            {
                throw new Exception($"Find duplicated argument name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", argNames)}");
            }
            var auxNames = symbol.ListAuxiliaryStates();
            var auxNamesDuplicate = auxNames.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));

            foreach (var name in auxNamesDuplicate)
            {
                throw new Exception($"Find duplicated auxiliary name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", argNames)}");
            }
        }


        public static void SaveCheckpoint(string prefix,
            int? epoch,
            Symbol symbol,
            Dictionary<string, NdArray> argParams,
            Dictionary<string, NdArray> auxParams)
        {
            symbol.Save($"{prefix}-symbol.json");

            Dictionary<string, NdArray> dict = new Dictionary<string, NdArray>();
            foreach (var kv in argParams)
            {
                dict.Add($"arg:{kv.Key}" , kv.Value);
            }
            foreach (var kv in auxParams)
            {
                dict.Add($"aux:{kv.Key}", kv.Value);
            }
            var paramName = $"{prefix}-{epoch:D4}.params";
            NdArray.Save(paramName, dict);
            ILog log = LogManager.GetLogger("");
            log.Info($"Saved checkpoint to \"{paramName}\"");
        }

        public static void LoadCheckpoint(string prefix,
           int? epoch,
          out Symbol symbol,
          out Dictionary<string, NdArray> argParams,
          out Dictionary<string, NdArray> auxParams)
        {
            symbol = Symbol.Load($"{prefix}-symbol.json");
            Dictionary<string, NdArray> saveDict;

            if (epoch.HasValue)
            {
                NdArray.Load($"{prefix}-{epoch:D4}.params", out saveDict);
            }
            else
            {
                int findindex = 0;
                int i = 0;
                for (; i < 10000; i++)
                {
                    var path = $"{prefix}-{i:D4}.params";
                    if (File.Exists(path))
                    {
                        findindex = i;
                    }
                }
                var readpath = $"{prefix}-{findindex:D4}.params";
                if (File.Exists(readpath))
                {
                    NdArray.Load(readpath, out saveDict);
                }
                else
                {
                    throw new FileNotFoundException("can not laod params files,check you prefix and epoch");
                }
            }

            argParams = new Dictionary<string, NdArray>();
            auxParams = new Dictionary<string, NdArray>();

            foreach (var kvitem in saveDict)
            {
                var sp = kvitem.Key.Split(':');
                var tp = sp[0];
                var name = sp[1];

                if (tp == "arg")
                {
                    argParams[name] = kvitem.Value;}
                if (tp == "aux")
                {
                    auxParams[name] = kvitem.Value;
                }
            }
        }
    }
}
