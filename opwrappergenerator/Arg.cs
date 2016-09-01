using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class Arg
    {
        private Dictionary<string, string> _type_dict = new Dictionary<string, string>
        {
            {"boolean", "bool"},
            {"Shape(tuple)", "Shape"},
            {"Symbol", "Symbol"},
            {"Symbol[]", "Symbol[]"},
            {"float", "float"},
            {"int", "int"},
            {"long", "long"},
            {"string", "string"}
        };
        public string orgin_name { get; }
        public string name { get; }
        public string description { get; }
        public bool is_enum { get; } = false;
        public EnumType Enum { get; }
        public string type_name { get; }
        public bool has_default { get; } = false;
        public string default_string { get; }
        public string default_string_with_object { get; } = "";

        public Arg(string op_name = "", string arg_name = "", string type_string = "", string desc_string = "")
        {
            if (arg_name == "src")
            {
                arg_name = "data";
            }
            this.orgin_name = arg_name;
            this.name = GetName(arg_name);
            this.description = desc_string;
            if (type_string.StartsWith("{"))
            {
                is_enum = true;
                Enum = new EnumType(op_name +"_" + arg_name, type_string);
                type_name = Enum.name;
            }
            else
            {
                string typename;

                if (_type_dict.TryGetValue(type_string.Split(' ').First().Replace(",", ""), out typename))
                {
                    type_name = typename;
                }
                else
                {
                    if (op_name == "Reshape")
                    {
                        type_name = "Shape";
                    }
                }

             
            }
            if (type_string.IndexOf("default", StringComparison.Ordinal) != -1)
            {
                has_default = true;
                default_string = type_string.Split(new string[] { "default=" }, StringSplitOptions.None)[1].Trim()
                    .Trim('\'');

                if (is_enum)
                {
                    default_string = Enum.GetDefaultValueString(default_string);
                }
                else if (default_string == "False")
                {
                    default_string = "false";
                }
                else if (default_string == "True")
                {
                    default_string = "true";
                }
                else if (default_string.StartsWith("("))
                {
                    if (default_string != "()")
                    {
                        default_string_with_object = $"if({name}==null){{ {name}= new Shape{default_string};}}\n";
                    }
     
                    default_string = "null";       
                }
                if (type_name == "float")
                {
                    default_string = default_string + "f";
                }
            }
            if (arg_name == "weight" || arg_name == "bias")
            {
                has_default = true;
                default_string = "null";
            }

        }

        private string GetName(string arg_name)
        {
            CultureInfo culture_info = Thread.CurrentThread.CurrentCulture;
            TextInfo text_info = culture_info.TextInfo;

            var namesp = arg_name.Split('_');


            return namesp.First()+ string.Join("", namesp.Skip(1).Select(s => text_info.ToTitleCase(s)));
        }
    }
    
}
