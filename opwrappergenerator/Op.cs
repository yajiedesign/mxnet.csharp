using System;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Threading;

namespace opwrappergenerator
{
    internal class Op
    {
        private static readonly Regex R = new Regex(@"
                (?<=[A-Z])(?=[A-Z][a-z]) |
                 (?<=[^A-Z])(?=[A-Z]) |
                 (?<=[A-Za-z])(?=[^A-Za-z])", RegexOptions.IgnorePatternWhitespace| RegexOptions.Compiled);

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
        enum MyEnum
        { 
            
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="use_name"></param>
        /// <returns></returns>
        public string GetOpDefinitionString(bool use_name)
        {
  
            string ret = "";
            var args_local = this._args.Skip(use_name ? 0 : 1).ToList();


            //enum
            if (use_name)
            {
                foreach (var arg in args_local.Where(w => w.is_enum))
                {
                    ret += $"/// <summary>\n/// {arg.description.Replace("\n","")}\n/// </summary>\n";
                    ret += arg.Enum.GetDefinitionString() +"\n";
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


            ret += $"public static Symbol {ConvertName(_name)}(";
            foreach (var arg in args_local)
            {
                ret += $"{arg.type_name} {arg.name}";
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
                ret += arg.default_string_with_object ;
            }

            ret += $"\nreturn new Operator(\"{_name}\")\n";

            foreach (var arg in _args)
            {
                if (arg.type_name == "Symbol" ||
                    arg.type_name == "Symbol[]" ||
                    arg.name == "symbolName")
                {
                    continue;
                }

                if (arg.is_enum)
                {
                    ret += $".SetParam(\"{arg.orgin_name}\", {arg.Enum.name}Convert[(int){arg.name}])\n";
                }
                else
                {
                    ret += $".SetParam(\"{arg.orgin_name}\", {arg.name})\n";
                }
        

            }


            foreach (var arg in _args)
            {
                if (arg.type_name != "Symbol")
                {
                    continue;
                }
                ret += $".SetInput(\"{arg.orgin_name}\", {arg.name})\n";
            }

            foreach (var arg in _args)
            {
                if (arg.type_name != "Symbol[]")
                {
                    continue;
                }
                ret += $".AddInput({arg.name})\n";
            }
            if (use_name)
            {
                ret += ".CreateSymbol(symbolName);\n";
            }
            else
            {

                ret += ".CreateSymbol();\n";
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