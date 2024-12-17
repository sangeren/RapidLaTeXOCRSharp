using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestLatex
{
    internal class BPE
    {
        public BPEModel model { get; set; }
    }
    internal class BPEModel
    {
        public Dictionary<string, int> vocab { get; set; }
        public Dictionary<int, string>? ReVocab { get; set; }
    }

}
