#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>

void preprocess(const string &path, const string & output);
void mat(const string &path, const string &output);


void bind_pre(py::module &m)
{

    /*
    Binding fast_cd_viewer
    */
    m.def("preprocess", &preprocess);
}
void bind_matfp(py::module &m)
{
    m.def("medial_axis_transform", &mat);
}

PYBIND11_MODULE(pymatfp, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: shaysweep

        .. autosummary::
           :toctree: _generate


    )pbdoc";

    bind_pre(m);
    bind_matfp(m);
}