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

        public Op(string name, string description, List<Arg> args)
        {
            this._name = name;
            this._description = description;

            var name_arg = new Arg(name,
                "symbol_name",
                "string",
                "name of the resulting symbol");
            args.Insert(0, name_arg);
            this._args = args.Where(w => !w.has_default).Concat(args.Where(w => w.has_default)).ToList();

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="use_name"></param>
        /// <param name="opNdArrayOrSymbol"></param>
        /// <returns></returns>
        public string GetOpDefinitionString(bool use_name, OpNdArrayOrSymbol opNdArrayOrSymbol,ref string enumret)
        {

            string NdArrayOrSymbol = "";
            switch (opNdArrayOrSymbol)
            {
                case OpNdArrayOrSymbol.Symbol:
                    NdArrayOrSymbol = "Symbol";
                    break;
                case OpNdArrayOrSymbol.NdArray:
                    NdArrayOrSymbol = "NdArray";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(opNdArrayOrSymbol), opNdArrayOrSymbol, null);
            }
            string ret = "";
            List<Arg> args_local = this._args.Skip(use_name ? 0 : 1).ToList();
            switch (opNdArrayOrSymbol)
            {
                case OpNdArrayOrSymbol.Symbol:
           
                    break;
                case OpNdArrayOrSymbol.NdArray:
                    if (use_name)
                    {
                        args_local = this._args.Skip(1).ToList();
                        args_local.Insert(0, new Arg("", "@out", "NDArray-or-Symbol", "output Ndarray") { });
                    }
              
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(opNdArrayOrSymbol), opNdArrayOrSymbol, null);
            }

            //enum
            if (use_name)
            {
                foreach (var arg in args_local.Where(w => w.is_enum))
                {
                    enumret += $"/// <summary>\n/// {arg.description.Replace("\n", "")}\n/// </summary>\n";
                    enumret += arg.Enum.GetDefinitionString() + "\n";
                    ret += arg.Enum.GetConvertString() + "\n";
                }
            }


            //comments 
            ret += $"/// <summary>\n/// {_description.Replace("\n", "")}\n/// </summary>\n";

            foreach (var arg in args_local)
            {
                ret += $"/// <param name=\"{arg.name}\">{arg.description.Replace("\n", "")}</param>\n";
            }
            ret += $" /// <returns>returns new symbol</returns>\n";

            ret += $"public static {NdArrayOrSymbol} {ConvertName(_name)}(";

            foreach (var arg in args_local)
            {
                if (arg.type_name == "NdArrayOrSymbol")
                {
                    ret += $"{NdArrayOrSymbol} {arg.name}";
                }
                else if (arg.type_name == "NdArrayOrSymbol[]")
                {
                    ret += $"{NdArrayOrSymbol}[] {arg.name}";
                }
                else
                {
                    ret += $"{arg.type_name} {arg.name}";
                }
                if (arg.has_default)
                {

                    ret += $"={arg.default_string}";
                }
                ret += ",\n";
            }
            if (args_local.Count > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }

            ret += ")\n{";

            foreach (var arg in args_local)
            {
                ret += arg.default_string_with_object;
            }

            ret += $"\nreturn new Operator(\"{_name}\")\n";

            foreach (var arg in _args)
            {
                if (arg.type_name == "NdArrayOrSymbol" ||
                    arg.type_name == "NdArrayOrSymbol[]" ||
                    arg.name == "symbol_name")
                {
                    continue;
                }

                if (arg.is_enum)
                {
                    ret += $".SetParam(\"{arg.orgin_name}\", Util.EnumToString<{arg.Enum.name}>({arg.name},{arg.Enum.name}Convert))\n";
                }
                else
                {
                    ret += $".SetParam(\"{arg.orgin_name}\", {arg.name})\n";
                }


            }


            foreach (var arg in _args)
            {
                if (arg.type_name != "NdArrayOrSymbol")
                {
                    continue;
                }
                ret += $".SetInput(\"{arg.orgin_name}\", {arg.name})\n";
            }

            foreach (var arg in _args)
            {
                if (arg.type_name != "NdArrayOrSymbol[]")
                {
                    continue;
                }
                ret += $".AddInput({arg.name})\n";
            }

            switch (opNdArrayOrSymbol)
            {
                case OpNdArrayOrSymbol.Symbol:
                {
                    if (use_name)
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
                    if (use_name)
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
            CultureInfo culture_info = Thread.CurrentThread.CurrentCulture;
            TextInfo text_info = culture_info.TextInfo;



            var ret = R.Replace(name, "_");
            return text_info.ToTitleCase(ret).Replace("_", "");
        }
    }
}