#include <string>
#include <iostream>
using namespace std;

void preprocess(const string &path, const string & output);
void mat(const string &path, const string &output);

int main(int argc, char *argv[]) {
    string input = argv[1];
    string output = argv[2];
    string basename = input.substr(0, input.find_last_of("."));
    string tmp = basename + "_tmp.geogram";
    preprocess(input, tmp);
    mat(tmp, output);
    cout << "Done converting " << input << " to " << output << ", tmp file saved at " << tmp << endl;

    
    return 0;
}