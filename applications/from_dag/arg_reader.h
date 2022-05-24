#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>

using std::vector;
using std::string;
using std::unordered_map;

struct arg_reader_t {
  arg_reader_t() {}

  void register_int(   string name, int         default_value) { ints[name]    = default_value; }
  void register_float( string name, float       default_value) { floats[name]  = default_value; }
  void register_vec(   string name, vector<int> default_value) { vecs[name]    = default_value; }
  void register_string(string name, string      default_value) { strings[name] = default_value; }

  void read(int argc, char** argv);

  int&         get_int(   string name){ return ints.at(name);    }
  float&       get_float( string name){ return floats.at(name);  }
  vector<int>& get_vec(   string name){ return vecs.at(name);    }
  string&      get_string(string name){ return strings.at(name); }

private:
  unordered_map<string, int> ints;
  unordered_map<string, float> floats;
  unordered_map<string, vector<int>> vecs;
  unordered_map<string, string> strings;

  static int read_int(int argc, char** argv, int& ret);
  static int read_float(int argc, char** argv, float& ret);
  static int read_vec(int argc, char** argv, vector<int>& ret);
  static int read_string(int argc, char** argv, string& ret);
};

int arg_reader_t::read_int(int argc, char** argv, int& ret)
{
  if(argc < 1) {
    return -1;
  }
  try {
    ret = std::stoi(std::string(argv[0]));
  } catch(std::invalid_argument const& ex) {
    return -1;
  } catch(std::out_of_range const& ex) {
    return -1;
  }
  return 1;
};

int arg_reader_t::read_float(int argc, char** argv, float& ret)
{
  if(argc < 1) {
    return -1;
  }
  try {
    ret = std::stof(std::string(argv[0]));
  } catch(std::invalid_argument const& ex) {
    return -1;
  } catch(std::out_of_range const& ex) {
    return -1;
  }
  return 1;
};

// Example -> Result
// [1,2,3,4,5] -> 1,2,3,4,5
// [1,2,3, 4,5] -> fail           (assuming arg split by whether or not there are spaces)
// [1,2.3] > 1,2,3.
// [1@$!$!2!$!#$!3] -> 1,2,3
int arg_reader_t::read_vec(int argc, char** argv, vector<int>& ret)
{
  vector<int> x;
  if(argc < 1) {
    return -1;
  }
  std::string s = argv[0];
  std::stringstream f(s);
  while(f && f.peek() != '[') {
    f.get();
  }
  if(!f) {
    return -1;
  }
  while(f) {
    char c = f.peek();
    if(std::isdigit(static_cast<unsigned char>(c))) {
      x.push_back(0);
      f >> x.back();
    } else
    if(c == ']') {
      ret = x;
      return 1;
    } else {
      f.get();
    }
  }
  // Nothing left in the string and did not hit ]...
  return -1;
};

int arg_reader_t::read_string(int argc, char** argv, string& ret) {
  if(argc < 1) {
    return -1; 
  }
  ret = std::string(argv[0]);
  return 1; 
}

void arg_reader_t::read(int argc, char** argv) {
  auto increment = [&](int n) {
    argc -= n;
    argv += n;
  };

  // discard the program name
  increment(1);

  while(argc != 0) {
    std::string matcher = argv[0];
    std::cout << matcher << ": ";
    increment(1);
    int n = -1;
    if(ints.count(matcher) > 0) {
      n = read_int(argc, argv, ints[matcher]);
      std::cout << ints[matcher];
    } else
    if(floats.count(matcher) > 0) {
      n = read_float(argc, argv, floats[matcher]);
      std::cout << floats[matcher];
    } else
    if(vecs.count(matcher) > 0) {
      n = read_vec(argc, argv, vecs[matcher]);
      std::cout << vecs[matcher];
    } else
    if(strings.count(matcher) > 0) {
      n = read_string(argc, argv, strings[matcher]);
      std::cout << strings[matcher];
    }
    if(n < 0) {
      throw std::runtime_error("could not parse args");
    }
    increment(n);

    std::cout << std::endl;
  }
}

//int main(int argc, char** argv) {
//  arg_reader_t reader;
//
//  reader.register_int("--n", 0);
//  reader.register_float("--m", 0.0);
//  reader.register_vec("--o", {});
//
//  reader.read(argc, argv);
//
//  std::cout << reader.get_int("--n") << std::endl;
//  std::cout << reader.get_float("--m") << std::endl;
//  std::cout << reader.get_vec("--o") << std::endl;
//}

