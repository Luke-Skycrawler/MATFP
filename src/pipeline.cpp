#include <string> 
#include <iostream>
#include "matfp/Logger.h"
#include "pre/pre_meshIO.h"
#include "pre/sharp_feature_detection.h"
#include "matfp/Args.h"

#include "matfp/AABBWrapper.h"
#include "matfp/Common.h"
#include "matfp/FeaturePreservation.h"
#include "matfp/InternalFeatureAddition.h"
#include "matfp/IterateSpheres.h"
#include "matfp/LFS.h"
#include "matfp/Logger.h"
#include "matfp/MedialMeshGenerator.h"
#include "matfp/MedialSpheresProcessor.h"
#include "matfp/MeshIO.h"
#include "matfp/MeshProcessor.h"
#include "matfp/NonManifoldMesh/Nonmanifoldmesh.h"
#include "matfp/RPDGenerator.h"
#include "matfp/ShrinkSpheres.h"
#include "matfp/Thinning.h"
#include "matfp/Triangulation.h"
#include "matfp/Types/CommonTypes.h"
#include "matfp/Types/OtherTypes.h"
#include "matfp/UpdateSpheres.h"
#include "matfp/WindingFilter.h"

using namespace std;
namespace py = pybind11;
using namespace matfp;
using namespace pre_matfp;

void preprocess(const string &path, const string &output_name)
{
    auto m_shape = new Shape3D();
    Args args;
    args.input_surface_path = path;
    GEO::initialize();
    //GEO::CmdLine::import_arg_group("algo");

    // Load mesh and detect features
    if (!load_mesh_and_preprocess(args, m_shape->sf_mesh))
    {
        logger().error("Unable to load mesh at {}", path);
        return;
    }

    if (args.is_save_model)
    {
        save_mesh(output_name, m_shape->sf_mesh);
    }
    GEO::terminate();
}

void mat(const string &path, const string &output)
{
    Args args;
    GEO::initialize();
    //GEO::CmdLine::import_arg_group("algo");

    args.input_surface_path = path;
    auto m_shape3D = new ThreeDimensionalShape;
    m_shape3D->mesh_name = path;

    // Load mesh
    if (!MeshIO::load_mesh_from_geogram(path, m_shape3D->sf_mesh,
                                        m_shape3D->sf_mesh_wrapper.input_vertices,
                                        m_shape3D->sf_mesh_wrapper.input_faces))
    {
        logger().error("Unable to load .geogram mesh at {}", path);
        return;
    }
    // Load sharp features
    matfp::load_mesh_features(
        m_shape3D->sf_mesh, m_shape3D->sf_mesh_wrapper.input_vertices,
        m_shape3D->sf_mesh_wrapper.input_faces, m_shape3D->s_edges,
        m_shape3D->se_normals, m_shape3D->se_ref_fs_pairs, m_shape3D->cc_edges,
        m_shape3D->corners, m_shape3D->sf_mesh_wrapper.conn_tris);
    matfp::init_input_faces_normals(m_shape3D->sf_mesh,
                                    m_shape3D->sf_mesh_wrapper.input_fnormals);
    // only load sharp edges, concave edges are not loaded
    // until load_concave_edges() is called
    m_shape3D->reload_ref_fs_pairs_not_cross();
    if (!m_shape3D->params.init(m_shape3D->get_sf_diag(),
                                args.downsample_percentage, args.rsample,
                                GEO::Geom::mesh_area(m_shape3D->sf_mesh)))
    {
        logger().error("Unable to init mesh parameters at {}", path);
        return;
    }

    // stage 1
    matfp::generate_lfs(*(m_shape3D), false /*is_debug*/);
    // show_LFS(m_shape3D);
    matfp::mesh_remesh_split_sharp_edges(*(m_shape3D),
                                         false /*is_debug*/);
    //show_init_seeds(m_shape3D);
    matfp::update_dt(*(m_shape3D), false /*is_debug*/);
    matfp::prepare_all_medial_spheres(*(m_shape3D),
                                      false /*is_debug*/);

    // it performs better when cc_len_eps is smaller, and we do not need
    // cc_normal_eps to be very small
    matfp::insert_spheres_for_concave_lines(
        m_shape3D->tan_cc_lines,
        m_shape3D->all_medial_spheres, args.cc_len_eps,
        args.cc_normal_eps);
    logger().info("Stage1 took {}s", 0.0);

    // stage 2
    matfp::shrink_spheres(
        m_shape3D->sf_mesh, m_shape3D->aabb_wrapper,
        m_shape3D->tan_cc_lines,
        m_shape3D->all_medial_spheres, false /*is_debug*/);
    matfp::update_spheres(m_shape3D->sf_mesh,
                          m_shape3D->ref_fs_pairs_not_cross,
                          m_shape3D->aabb_wrapper,
                          m_shape3D->all_medial_spheres,
                          false /*is_debug*/);
    logger().info("Stage2 took {}s", 0.0);

    // stage 3
    matfp::add_or_delete_for_se(
        m_shape3D->all_medial_spheres,
        m_shape3D->se_spheres,
        m_shape3D->se_kd_tree_idx_to_se_tag,
        m_shape3D->se_kd_tree, m_shape3D->se_kd_points,
        true /*is_check_updated_only*/,
        m_shape3D->params.is_using_dilated_radius,
        false /*is_debug*/);
    matfp::generate_RT_for_dual_PD(
        m_shape3D->all_medial_spheres,
        m_shape3D->valid_medial_spheres,
        m_shape3D->seed_points, m_shape3D->feature_points,
        m_shape3D->params.bb_points, m_shape3D->rt,
        m_shape3D->params.is_using_dilated_radius,
        false /*is_debug*/);
    matfp::generate_RT_dual_info(m_shape3D->sf_mesh_wrapper.VI,
                                 m_shape3D->sf_mesh_wrapper.FI,
                                 m_shape3D->sf_mesh,
                                 m_shape3D->aabb_wrapper,
                                 m_shape3D->rt, false /*is_debug*/);
    matfp::create_mat_from_RT(
        m_shape3D->all_medial_spheres,
        m_shape3D->valid_medial_spheres, m_shape3D->rt,
        m_shape3D->mesh_name, m_shape3D->mat_refined,
        m_shape3D->params.is_using_dilated_radius,
        false /*is_debug*/);
    /*show_mat_simple(m_shape3D->mat_refined,
                               RestrictedType::RPD);*/
    logger().info("Stage3 took {}s", 0.0);

    // stage 4
    for (int i = 0; i < 2; i++)
    {
        matfp::check_invalid_mat_edges(
            m_shape3D->mat_refined,
            m_shape3D->valid_medial_spheres,
            m_shape3D->all_medial_spheres,
            m_shape3D->mat_refined.invalid_mat_edges,
            true /*is_faster*/, false /*is_debug*/);
        // logger().info("Check_invalid took {}s", timer.getElapsedTimeInSec());
        matfp::insert_internal_feature_spheres(
            m_shape3D->sf_mesh,
            m_shape3D->ref_fs_pairs_not_cross,
            m_shape3D->aabb_wrapper,
            m_shape3D->mat_refined.invalid_mat_edges,
            m_shape3D->valid_medial_spheres,
            m_shape3D->all_medial_spheres, false /*is_debug*/);
        matfp::add_or_delete_for_se(
            m_shape3D->all_medial_spheres,
            m_shape3D->se_spheres,
            m_shape3D->se_kd_tree_idx_to_se_tag,
            m_shape3D->se_kd_tree, m_shape3D->se_kd_points,
            true /*is_check_updated_only*/,
            m_shape3D->params.is_using_dilated_radius,
            false /*is_debug*/);
        matfp::generate_RT_for_dual_PD(
            m_shape3D->all_medial_spheres,
            m_shape3D->valid_medial_spheres,
            m_shape3D->seed_points,
            m_shape3D->feature_points,
            m_shape3D->params.bb_points, m_shape3D->rt,
            m_shape3D->params.is_using_dilated_radius,
            false /*is_debug*/);
        matfp::generate_RT_dual_info(
            m_shape3D->sf_mesh_wrapper.VI,
            m_shape3D->sf_mesh_wrapper.FI,
            m_shape3D->sf_mesh, m_shape3D->aabb_wrapper,
            m_shape3D->rt, false /*is_debug*/);
        matfp::create_mat_from_RT(
            m_shape3D->all_medial_spheres,
            m_shape3D->valid_medial_spheres, m_shape3D->rt,
            m_shape3D->mesh_name, m_shape3D->mat_refined,
            m_shape3D->params.is_using_dilated_radius,
            false /*is_debug*/);
    }
    /*show_mat_simple(m_shape3D->mat_refined,
                               RestrictedType::RPD);*/
    logger().info("Stage4 took {}s", 0.0);

    // save .ply
    MatIO::write_nmm_ply(m_shape3D->mat_refined.mat_name,
                         m_shape3D->mat_refined);
    logger().info("Saved MAT to {}.ply",
                  m_shape3D->mat_refined.mat_name);

    // save .ma
    // MatIO::export_nmm(m_shape3D->mat_refined.mat_name,
    //   m_shape3D->mat_refined);
    MatIO::export_ma_clean(m_shape3D->mat_refined.mat_name,
                           m_shape3D->mat_refined);
    logger().info("Saved MAT to {}.ma",
                  m_shape3D->mat_refined.mat_name);

    GEO::terminate();
}