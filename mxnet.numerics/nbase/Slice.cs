using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public class Slice
    {
        public Slice()
        {
            this.start = 0;
            this.end = 0;
            this.step = 1;

        }
        public Slice(int start, int end, int step = 1)
        {
            if (step == 0)
            {
                throw new ArgumentException($"{nameof(step)} can not be equal to 0");
            }
            //if (step > 0)
            //{
            //    if (start > 0 && end > 0 && start >= end)
            //    {
            //        throw new ArgumentException($"{nameof(start)} must be greater than {nameof(end)} when {nameof(step)} is positive");
            //    }
            //    else  if (start < 0 && end < 0 && start >= end)
            //    {
            //        throw new ArgumentException($"{nameof(start)} must be greater than {nameof(end)} when {nameof(step)} is positive");
            //    }
            //}
            //else
            //{
            //    if (start > 0 && end > 0 && end >= start)
            //    {
            //        throw new ArgumentException($"{nameof(end)} must be greater than {nameof(start)} when {nameof(step)} is negative");
            //    }
            //    else if (start < 0 && end < 0 && end >= start)
            //    {
            //        throw new ArgumentException($"{nameof(end)} must be greater than {nameof(start)} when {nameof(step)} is negative");
            //    }
            //}

            this.start = start;
            this.end = end;
            this.step = step;
        }

        public int start { get; private set; }
        public int end { get; private set; }

        public int step { get; private set; }

        public int size
        {
            get
            {
                var temp = (int) Math.Ceiling((end - start)/(double) step);
                return temp > 0 ? temp : 0;
            }
        }

        /// <summary>
        /// transformed slice size absolute to relative
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public Slice Translate(uint dim)
        {
            int ret_start;
            int ret_end;
            int ret_step;
            if (step > 0)
            {
                ret_start = SetStart(start, dim , step);
                ret_end = SetEnd(end, dim, step);

                if (ret_end >= dim)
                {
                    ret_end = (int)dim;
                }
                ret_step = step;
            }
            else
            {
                ret_start = SetEnd(start, dim, step);
                ret_end = SetStart(end, dim, step);

                if (ret_start >= dim)
                {
                    ret_start = (int)dim - 1;
                }
                ret_step = step;
            }


            Slice ret = new Slice(ret_start, ret_end, ret_step);
            return ret;
        }
        private static int SetStart(int start, uint dim, int step)
        {
            if (start ==  int.MinValue)
            {
                start = 0;
            }
            else if (start ==  int.MaxValue)
            {
                start = -1;
            }
            else if (start < 0)
            {
                start = (int)dim + start;
            }

            return start;
        }

        private static int SetEnd(int end, uint dim, int step)
        {
            if (end ==  int.MaxValue)
            {
                end = (int)dim; 
            }
            else if (end == int.MinValue)
            {
                end = (int)dim;
            }
            else if (end < 0)
            {
                end = (int) dim + end;
            }
            return end;
        }



        private static readonly Regex RegNoStep = new Regex("(.*):(.*)");
        private static readonly Regex RegWithStep = new Regex("(.*):(.*):(.*)");
        public static implicit operator Slice(string slice)
        {
            int start = 0;
            if (int.TryParse(slice, out start))
            {
                return new Slice(start, start + 1, 1);
            }

            bool withstep = false;
            var m = RegWithStep.Match(slice);
            if (!m.Success)
            {
                m = RegNoStep.Match(slice);
            }
            else
            {
                withstep = true;
            }

            if (!m.Success) { throw new ArgumentException(nameof(slice));}
          
            string str_start = m.Groups[1].Value;
            if (!string.IsNullOrWhiteSpace(str_start) && !int.TryParse(str_start, out start))
            {
                throw new ArgumentException($"{nameof(start)} must be number");
            }
            int end = 0;
            string str_end = m.Groups[2].Value;
            if (!string.IsNullOrWhiteSpace(str_end) && !int.TryParse(str_end, out end))
            {
                throw new ArgumentException($"{nameof(end)} must be number");
            }
            int step = 1;
            if (withstep)
            {
                string str_step = m.Groups[3].Value;
                if (!string.IsNullOrWhiteSpace(str_step) && !int.TryParse(str_step, out step))
                {
                    throw new ArgumentException($"{nameof(step)} must be number");
                }
            }
            if (string.IsNullOrWhiteSpace(str_start))
            {
                start = int.MinValue;
            }
            if (string.IsNullOrWhiteSpace(str_end))
            {
                end = int.MaxValue;
            }
            return new Slice(start, end, step);
        }

        public Slice SubSlice(Slice sub)
        {
            int local_start = 0;
            int local_end = 0;
            int local_step = 0;
            if (this.step > 0 && sub.step > 0)
            {
                local_start = this.start + sub.start;
                local_end = this.start + sub.end;
                local_step = this.step * sub.step;
                if (local_start > this.end)
                {
                    local_start = this.end;
                }
                if (local_end > this.end)
                {
                    local_end = this.end;
                }
            }

            if (this.step > 0 && sub.step < 0)
            {
                local_start = this.start + sub.start;
                local_end = this.start + sub.end;
                local_step = this.step * sub.step;
                if (local_start >= this.end)
                {
                    local_start = this.end ;
                }
                if (local_end >= this.end)
                {
                    local_end = this.end;
                }
            }

            if (this.step < 0 && sub.step > 0)
            {
                local_start = this.start - sub.end;
                local_end = this.start - sub.start;
                local_step = this.step * sub.step;
                if (local_start < this.end)
                {
                    local_start = this.end;
                }
                if (local_end < this.end)
                {
                    local_end = this.end;
                }
            }


            if (this.step < 0 && sub.step < 0)
            {
                local_end = this.start - sub.start;
                local_start = this.start - sub.end;
                local_step = this.step * sub.step;
                if (local_start <= this.end)
                {
                    local_start = this.end;
                }
            }

       
      
            return new Slice(local_start, local_end, local_step);
        }

        public static Slice[] FromShape(uint[] data)
        {
            return data.Select(s => new Slice(0, (int) s, 1)).ToArray();
        }
    }
}
