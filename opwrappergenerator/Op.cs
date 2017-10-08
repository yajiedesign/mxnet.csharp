using System;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Threading;

namespace opwrappergenerator
{    /// <summary>
     /// 
     /// </summary>
    enum OpNdArrayOrSymbol
    {
        Symbol,
        NdArray


    }
    internal class Op
    {
        private static readonly Regex R = new Regex(@"
                (?<=[A-Z])(?=[A-Z][a-z]) |
                 (?<=[^A-Z])(?=[A-Z]) |
                 (?<=[A-Za-z])(?=[^A-Za-z])", RegexOptions.IgnorePatternWhitespace | RegexOptions.Compiled);

        private readonly string _name;
        private readonly string _description;
        private readonly List<Arg> _args;
        private readonly List<Arg> _argCss;

        public Op(string name, string description, List<Arg> args)
        {
            this._name = name;
            this._description = description;

            var nameArg = new Arg(name,
                "symbol_name",
                "string",
                "name of the resulting symbol");
            args.Insert(0, nameArg);
            this._args = args.ToList();
            this._argCss = args.Where(w => !w.HasDefault).Concat(args.Where(w => w.HasDefault)).ToList();

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="useName"></param>
        /// <param name="opNdArrayOrSymbol"></param>
        /// <returns></returns>
        public string GetOpDefinitionString(bool useName, OpNdArrayOrSymbol opNdArrayOrSymbol,ref string enumret)
        {

            string ndArrayOrSymbol = "";
            switch (opNdArrayOrSymbol)
            {
                case OpNdArrayOrSymbol.Symbol:
                    ndArrayOrSymbol = "Symbol";
                    break;
                case OpNdArrayOrSymbol.NdArray:
                    ndArrayOrSymbol = "NdArray";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(opNdArrayOrSymbol), opNdArrayOrSymbol, null);
            }
            string ret = "";
            List<Arg> argsLocal = this._argCss.Skip(useName ? 0 : 1).ToList();
            switch (opNdArrayOrSymbol)
            {
                case OpNdArrayOrSymbol.Symbol:
           
                    break;
                case OpNdArrayOrSymbol.NdArray:
                    if (useName)
                    {
                        argsLocal = this._argCss.Skip(1).ToList();
                        argsLocal.Insert(0, new Arg("", "@out", "NDArray-or-Symbol", "output Ndarray") { });
                    }
              
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(opNdArrayOrSymbol), opNdArrayOrSymbol, null);
            }

            //enum
            if (useName)
            {
                foreach (var arg in argsLocal.Where(w => w.IsEnum))
                {
                    enumret += $"/// <summary>\n/// {arg.Description.Replace("\n", "")}\n/// </summary>\n";
                    enumret += arg.Enum.GetDefinitionString() + "\n";
                    ret += arg.Enum.GetConvertString() + "\n";
                }
            }


            //comments 
            ret += $"/// <summary>\n/// {_description.Replace("\n", "")}\n/// </summary>\n";

            foreach (var arg in argsLocal)
            {
                ret += $"/// <param name=\"{arg.Name}\">{arg.Description.Replace("\n", "")}</param>\n";
            }
            ret += $" /// <returns>returns new symbol</returns>\n";

            ret += $"public static {ndArrayOrSymbol} {ConvertName(_name)}(";

            foreach (var arg in argsLocal)
            {
                if (arg.TypeName == "NdArrayOrSymbol")
                {
                    ret += $"{ndArrayOrSymbol} {arg.Name}";
                }
                else if (arg.TypeName == "NdArrayOrSymbol[]")
                {
                    ret += $"{ndArrayOrSymbol}[] {arg.Name}";
                }
                else
                {
                    ret += $"{arg.TypeName} {arg.Name}";
                }
                if (arg.HasDefault)
                {

                    ret += $"={arg.DefaultString}";
                }
                ret += ",\n";
            }
            if (argsLocal.Count > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }

            ret += ")\n{";

            foreach (var arg in argsLocal)
            {
                ret += arg.DefaultStringWithObject;
            }

            ret += $"\nreturn new Operator(\"{_name}\")\n";

            foreach (var arg in _args)
            {
                if (arg.TypeName == "NdArrayOrSymbol" ||
                    arg.TypeName == "NdArrayOrSymbol[]" ||
                    arg.Name == "symbol_name")
                {
                    continue;
                }

                if (arg.IsEnum)
                {
                    ret += $".SetParam(\"{arg.OrginName}\", Util.EnumToString<{arg.Enum.Name}>({arg.Name},{arg.Enum.Name}Convert))\n";
                }
                else
                {
                    ret += $".SetParam(\"{arg.OrginName}\", {arg.Name})\n";
                }


            }


            foreach (var arg in _args)
            {
                if (arg.TypeName != "NdArrayOrSymbol")
                {
                    continue;
                }
                ret += $".SetInput(\"{arg.OrginName}\", {arg.Name})\n";
            }

            foreach (var arg in _args)
            {
                if (arg.TypeName != "NdArrayOrSymbol[]")
                {
                    continue;
                }
                ret += $".AddInput({arg.Name})\n";
            }

            switch (opNdArrayOrSymbol)
            {
                case OpNdArrayOrSymbol.Symbol:
                {
                    if (useName)
                    {
                        ret += ".CreateSymbol(symbol_name);\n";
                    }
                    else
                    {
                        ret += ".CreateSymbol();\n";
                    }
                   }
                    break;
                case OpNdArrayOrSymbol.NdArray:
                    if (useName)
                    {
                        ret += ".Invoke(@out);\n";
                    }
                    else
                    {
                        ret += ".Invoke();\n";
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(opNdArrayOrSymbol), opNdArrayOrSymbol, null);
            }
    
            ret += "}";
            return ret;
        }

        private string ConvertName(string name)
        {
            CultureInfo cultureInfo = Thread.CurrentThread.CurrentCulture;
            TextInfo textInfo = cultureInfo.TextInfo;



            var ret = R.Replace(name, "_");
            return textInfo.ToTitleCase(ret).Replace("_", "");
        }
    }
}