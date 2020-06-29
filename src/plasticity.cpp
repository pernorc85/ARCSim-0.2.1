/*
  Copyright ?2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.
  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.
  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "plasticity.h"

//#include "bah.hpp"
#include "types3D.h"
#include "vectors.h"
#include "optimization.h"
#include "cloth_physics.h"
#include <omp.h>

using namespace std;

static const double mu = 1e-6;

Mat2x2 edges_to_face (DCEL &mesh, const Vec3 &theta, const Face *face);
Vec3 face_to_edges (DCEL &mesh, const Mat2x2 &S, const Face *face);

void reset_plasticity (Cloth &cloth) {
    DCEL &mesh = cloth.mesh;
    for (int n = 0; n < mesh.mNodes.size(); n++)
        //in ARCSim y means plastic_embedding, which is also a size-3 vector.
        mesh.mNodes[n].pos_in_plastic_space = mesh.mNodes[n].pos;
    for (int e = 0; e < mesh.mEdges.size(); e++) {
        HalfEdgePtr edge = mesh.mEdges[e];
        edge->theta_ideal = mesh.GetDihedralAngle(edge);//???
        edge->damage = 0;
    }
    for (Face *face : mesh.mFaces) {
        double theta1 = mesh.GetDihedralAngle(face->edge);
        double theta2 = mesh.GetDihedralAngle(face->edge->nextEdge);
        double theta3 = mesh.GetDihedralAngle(face->edge->prevEdge); 
        Vec3 thetas = Vec3(theta1, theta2, theta3);
        face->S_plastic = edges_to_face(mesh, thetas, face);
        face->damage = 0;
    }
}

void recompute_edge_plasticity (DCEL &mesh);

void optimize_plastic_embedding (Cloth &cloth);

double norm_F(Mat2x2 A){
    double sum_sq = 0.0;
    sum_sq += A(0,0) * A(0,0);
    sum_sq += A(0,1) * A(0,1);
    sum_sq += A(1,0) * A(1,0);
    sum_sq += A(1,1) * A(1,1);
    return sqrt(sum_sq);
}

void plastic_update (Cloth &cloth) {
    DCEL &mesh = cloth.mesh;
    const vector<Cloth::Material*> &materials = cloth.materials;
    for (Face *face : mesh.mFaces) {
        double S_yield = materials[face->label]->yield_curv;
        double theta1 = mesh.GetDihedralAngle(face->edge);
        double theta2 = mesh.GetDihedralAngle(face->edge->nextEdge);
        double theta3 = mesh.GetDihedralAngle(face->edge->prevEdge);
        Vec3 thetas = Vec3(theta1, theta2, theta3);
        Mat2x2 S_total = edges_to_face(mesh, thetas, face);
        Mat2x2 S_elastic = S_total - face->S_plastic;
        double dS = norm_F(S_elastic);//Frobenius norm
        cout << "norm of S_elastic = " << dS << endl;
        if (dS > S_yield) {
            face->S_plastic += S_elastic/dS*(dS - S_yield);
            face->damage += dS/S_yield - 1;
            cout << "damage = " << face->damage << endl;
        }
    }
    int tmp;
    cin >> tmp;
    recompute_edge_plasticity(cloth.mesh);
}

// ------------------------------------------------------------------ //
//In different contex, thetas has different meaning.
//So it must use the parameter passed in
//face_edges must be ordered as [e, next e, next next e]
Mat2x2 edges_to_face (DCEL &mesh, const Vec3 &thetas, const Face *face) {
    Mat2x2 S;
    vector<Vertex> vs;
    vector<HalfEdgePtr> face_edges;
 
    HalfEdgePtr e = face->edge;
    vs.push_back(mesh.mVertices[e->origin]);
    face_edges.push_back(e);
    e = e->nextEdge;
    vs.push_back(mesh.mVertices[e->origin]);
    face_edges.push_back(e);
    e = e->nextEdge;
    vs.push_back(mesh.mVertices[e->origin]);
    face_edges.push_back(e);
    for (int i = 0; i < 3; i++) {
        Vec2 duv_edge = vs[(i+2)%3].uv - vs[(i+1)%3].uv;
        Vec2 t_mat = perp(normalize(duv_edge));
        double edge_length = norm(duv_edge);
        S -= 1/2. * thetas[i]* edge_length * outer(t_mat, t_mat);
    }
    S /= face->area;
    return S;
}

//per face plastic part of bending strain --> rest angle of each edge
Vec3 face_to_edges (DCEL &mesh, const Mat2x2 &S, const Face *face) {
    Vec3 s_tmp(S(0,0), S(1,1), S(0,1));
    Vec3 s = face->area*s_tmp;
    Mat3x3 A;
 
    vector<Vertex> vs;
    HalfEdgePtr e = face->edge;
    vs.push_back(mesh.mVertices[e->origin]);
    e = e->nextEdge;
    vs.push_back(mesh.mVertices[e->origin]);
    e = e->nextEdge;
    vs.push_back(mesh.mVertices[e->origin]);
    for (int i = 0; i < 3; i++) {
        Vec2 duv_edge = vs[(i+2)%3].uv - vs[(i+1)%3].uv;
        Vec2 t_mat = perp(normalize(duv_edge));
        Mat2x2 Se = -1/2. * norm(duv_edge) * outer(t_mat, t_mat);
        //A.col(i) = Vec3(Se(0,0), Se(1,1), Se(0,1));
        A.Set(0, i, Se(0,0));
        A.Set(1, i, Se(1,1));
        A.Set(2, i, Se(0,1));
    }
    return inverse(A)*s;
}

void recompute_edge_plasticity (DCEL &mesh) {
    for (HalfEdgePtr edge : mesh.mEdges) {
        edge->theta_ideal = 0;
        edge->damage = 0;
    }
    for (int f = 0; f < mesh.mFaces.size(); f++) {
        const Face *face = mesh.mFaces[f];
        Vec3 ideal_thetas = face_to_edges(mesh, face->S_plastic, face);
        vector<HalfEdgePtr> face_edges;
        face_edges.push_back(face->edge);
        face_edges.push_back(face->edge->nextEdge);
        face_edges.push_back(face->edge->nextEdge->nextEdge); 
        for (int e = 0; e < 3; e++) {
            face_edges[e]->theta_ideal += ideal_thetas[e];
            face_edges[e]->damage += face->damage;
        }
    }
    int tmp; cin >> tmp;

    for (HalfEdgePtr edge : mesh.mEdges) {
        if (edge->twinEdge) {// edge has two adjacent faces
            edge->theta_ideal /= 2.0;
            edge->damage /= 2.0;
            cout << "edge theta_ideal = " << edge->theta_ideal 
                 << "edge damage = " << edge->damage << endl;
        }
        //edge->reference_angle = edge->theta_ideal;
    }
}

// ------------------------------------------------------------------ //

struct EmbedOpt: public NLOpt {
    Cloth &cloth;
    DCEL &mesh;
    vector<Vec3> y0;
    mutable vector<double> f;
    SparseMatrix<double> K;
    map<NodeId, int> nid_to_matpos_map;
    map<int, NodeId> matpos_to_nid_map;

    EmbedOpt (Cloth &cloth): cloth(cloth), mesh(cloth.mesh) {
        int matpos = 0;
        for (auto &item : mesh.mNodes) {
            NodeId nid = item.first; 
            nid_to_matpos_map[nid] = matpos;
            matpos_to_nid_map[matpos] = nid;
            matpos += 3;
        }

        int nn = mesh.mNodes.size();
        nvar = nn*3;
        y0.resize(nn);
        for (auto &item : mesh.mNodes) {
            NodeId nid = item.first;
            int matpos = nid_to_matpos_map[nid];
            mesh.mNodes[nid].pos_in_plastic_space = mesh.mNodes[nid].pos;
            y0[matpos/3] = mesh.mNodes[nid].pos_in_plastic_space;
        }
        f.resize(nvar);
        K = SparseMatrix<double>(nn*3,nn*3);
    }
    void initialize (double *x) const;
    virtual void precompute (const double *x) ;
    double objective (const double *x) const;
    void gradient (const vector<double> &x, vector<double> &g) const;
    virtual bool hessian (const double *x, SparseMatrix<double> &H) const;
    void finalize (const double *x) const;
};

void reduce_stretching_stiffnesses (vector<Cloth::Material*> &materials);
void restore_stretching_stiffnesses (vector<Cloth::Material*> &materials);


void optimize_plastic_embedding (Cloth &cloth) {
    // vector<Cloth::Material> materials = cloth.materials;
    reduce_stretching_stiffnesses(cloth.materials);
    NLOpt *problem = new EmbedOpt(cloth);
    line_search_newtons_method(problem, OptOptions().max_iter(5), true);
    restore_stretching_stiffnesses(cloth.materials);
    // cloth.materials = materials;
}



void EmbedOpt::initialize (double *x) const {
    for (int i = 0; i < 3 * mesh.mNodes.size(); i++)
        x[i] = 0.0;
}

void EmbedOpt::precompute (const double *x) {
    int nn = mesh.mNodes.size();
    f.resize(nn*3);
    cout << "EmbedOpt precompute" << endl;
    K = SparseMatrix<double>(nn*3, nn*3);
    for (auto &item : mesh.mNodes) {
        NodeId nid = item.first;
        int matpos = nid_to_matpos_map.at(nid); 
        //in ARCSim, y means plastic embedding, which is also Vec3
        mesh.mNodes[nid].pos_in_plastic_space.x = y0[matpos/3](0) + x[matpos];
        mesh.mNodes[nid].pos_in_plastic_space.y = y0[matpos/3](1) + x[matpos+1];
        mesh.mNodes[nid].pos_in_plastic_space.z = y0[matpos/3](2) + x[matpos+2];
    }
    bool plastic_space = true;
    add_internal_forces(cloth.mesh, cloth.materials, K, f, 0, nid_to_matpos_map, plastic_space);
}

double EmbedOpt::objective (const double *x) const {
    for (auto &item : mesh.mNodes){
        NodeId nid = item.first;
        int matpos = nid_to_matpos_map.at(nid);

        mesh.mNodes[nid].pos_in_plastic_space.x = y0[matpos/3](0) + x[matpos];
        mesh.mNodes[nid].pos_in_plastic_space.y = y0[matpos/3](1) + x[matpos+1];
        mesh.mNodes[nid].pos_in_plastic_space.z = y0[matpos/3](2) + x[matpos+2];
    }
    bool plastic_space = true;
    return internal_energy(cloth.mesh, cloth.materials, plastic_space);
}

void EmbedOpt::gradient (const vector<double> &x, vector<double> &g) const {
    cout << "EmbedOpt gradient" << endl;
    for (int i = 0; i < 3*mesh.mNodes.size(); i++) {
        g[i] = -f[i];
    }
}

static Mat3x3 get_submat (SparseMatrix<double> &A, int i, int j) {
    Mat3x3 Aij;
    for (int ii = 0; ii < 3; ii++) 
        for (int jj = 0; jj < 3; jj++)
            Aij.Set(ii,jj, A(i*3+ii, j*3+jj));
    return Aij;
}
static void set_submat (SparseMatrix<double> &A, int i, int j, const Mat3x3 &Aij) {
    for (int ii = 0; ii < 3; ii++) 
        for (int jj = 0; jj < 3; jj++)
            A.Insert(i*3+ii, j*3+jj, Aij(ii,jj));
}

static void add_submat (SparseMatrix<double> &A, int i, int j, const Mat3x3 &Aij) {
    for (int ii = 0; ii < 3; ii++) 
        for (int jj = 0; jj < 3; jj++)
            A.Add(i*3+ii, j*3+jj, Aij(ii,jj));
}

bool EmbedOpt::hessian (const double *x, SparseMatrix<double> &H) const {
    H = K;
    for (int i = 0; i < mesh.mNodes.size(); i++) {
        add_submat(H, i, i, Mat3x3(::mu));
    }
    return true;
}

void EmbedOpt::finalize (const double *x) const {
    for (auto &item : mesh.mNodes) {
        NodeId nid = item.first;
        int matpos = nid_to_matpos_map.at(nid);
        item.second.pos_in_plastic_space.x = y0[matpos/3](0) + x[matpos];
        item.second.pos_in_plastic_space.y = y0[matpos/3](1) + x[matpos+1];
        item.second.pos_in_plastic_space.z = y0[matpos/3](2) + x[matpos+2];
    }
}

void reduce_stretching_stiffnesses (vector<Cloth::Material*> &materials) {
    for (int m = 0; m < materials.size(); m++)
        for (int i = 0; i < 40; i++)
            for (int j = 0; j < 40; j++)
                for (int k = 0; k < 40; k++)
                    materials[m]->stretching.s[i][j][k] *= 1e-2;
}

void restore_stretching_stiffnesses (vector<Cloth::Material*> &materials) {
    for (int m = 0; m < materials.size(); m++)
        for (int i = 0; i < 40; i++)
            for (int j = 0; j < 40; j++)
                for (int k = 0; k < 40; k++)
                    materials[m]->stretching.s[i][j][k] *= 1e2;
}

// ------------------------------------------------------------------ //


const vector<Residual> *res_old;

void resample_callback (Face *face_new, const Face *face_old);

/*
vector<Residual> back_up_residuals (DCEL &mesh) {
    vector<Residual> res(mesh.mFaces.size());
    bool plastic_space_flag = true;
    for (int f = 0; f < mesh.mFaces.size(); f++) {
        const Face *face = mesh.mFaces[f];
      
        vector<HalfEdgePtr> face_edges;
        HalfEdgePtr e = face->edge;
        face_edges.push_back(e);
        e = e->nextEdge;
        face_edges.push_back(e);
        e = e->nextEdge;
        face_edges.push_back(e);
      
        Vec3 thetas_residual;
        for (int i = 0; i < 3; i++) {
            const HalfEdgePtr edge = face_edges[i];
            thetas_residual[i] = edge->theta_ideal - mesh.GetDihedralAngle(edge, plastic_space_flag);
        }
        res[f].S_res = edges_to_face(mesh, thetas_residual, face);
        res[f].damage = face->damage;
    }
    return res;
}

void restore_residuals (DCEL &mesh, const DCEL &old_mesh,
                        const vector<Residual> &res_old) {
    ::res_old = &res_old;
    BahNode *tree = new_bah_tree(old_mesh);
    bool plastic_space_flag = true;
#pragma omp parallel for
    for (int f = 0; f < mesh.mFaces.size(); f++) {
        Face *face = mesh.mFaces[f];
        vector<HalfEdgePtr> face_edges;
        HalfEdgePtr e = face->edge;
        face_edges.push_back(e);
        e = e->nextEdge;
        face_edges.push_back(e);
        e = e->nextEdge;
        face_edges.push_back(e);
        
        Vec3 theta;
        for (int i = 0; i < 3; i++)
            theta[i] = mesh.GetDihedralAngle(face_edges[i], plastic_space_flag);
        face->S_plastic = edges_to_face(mesh, theta, face);
        face->damage = 0;
        for_overlapping_faces(face, tree, resample_callback);
    }
    delete_bah_tree(tree);
    recompute_edge_plasticity(mesh);
}


double overlap_area (const Face *face0, const Face *face1);
void resample_callback (Face *face_new, const Face *face_old) {
    double a = overlap_area(face_new, face_old)/face_new->a;
    const Residual &res = (*::res_old)[face_old->index];
    face_new->S_plastic += a*res.S_res;
    face_new->damage += a*res.damage;
}
// ------------------------------------------------------------------ //
vector<Vec2> sutherland_hodgman (const vector<Vec2> &poly0,
                                 const vector<Vec2> &poly1);
double area (const vector<Vec2> &poly);
double overlap_area (const Face *face0, const Face *face1) {
    vector<Vec2> u0(3), u1(3);
    Vec2 u0min(face0->v[0]->u), u0max(u0min),
         u1min(face1->v[0]->u), u1max(u1min);
    for (int i = 0; i < 3; i++) {
        u0[i] = face0->v[i]->u;
        u1[i] = face1->v[i]->u;
        u0min = vec_min(u0min, u0[i]);
        u0max = vec_max(u0max, u0[i]);
        u1min = vec_min(u1min, u1[i]);
        u1max = vec_max(u1max, u1[i]);
    }
    if (u0min[0] > u1max[0] || u0max[0] < u1min[0]
        || u0min[1] > u1max[1] || u0max[1] < u1min[1]) {
        return 0;
    }
    return area(sutherland_hodgman(u0, u1));
}
vector<Vec2> clip (const vector<Vec2> &poly, const Vec2 &clip0,
                                             const Vec2 &clip1);
vector<Vec2> sutherland_hodgman (const vector<Vec2> &poly0,
                                 const vector<Vec2> &poly1) {
    vector<Vec2> out(poly0);
    for (int i = 0; i < 3; i++)
        out = clip(out, poly1[i], poly1[(i+1)%poly1.size()]);
    return out;
}
double distance (const Vec2 &v, const Vec2 &v0, const Vec2 &v1) {
    return wedge(v1-v0, v-v0);}
Vec2 lerp (double t, const Vec2 &a, const Vec2 &b) {return a + t*(b-a);}
vector<Vec2> clip (const vector<Vec2> &poly, const Vec2 &clip0,
                                             const Vec2 &clip1) {
    if (poly.empty())
        return poly;
    vector<Vec2> newpoly;
    for (int i = 0; i < poly.size(); i++) {
        const Vec2 &v0 = poly[i], &v1 = poly[(i+1)%poly.size()];
        double d0 = distance(v0, clip0, clip1), d1 = distance(v1, clip0, clip1);
        if (d0 >= 0)
            newpoly.push_back(v0);
        if (!((d0<0 && d1<0) || (d0==0 && d1==0) || (d0>0 && d1>0)))
            newpoly.push_back(lerp(d0/(d0-d1), v0, v1));
    }
    return newpoly;
}
double area (const vector<Vec2> &poly) {
    if (poly.empty())
        return 0;
    double a = 0;
    for (int i = 1; i < poly.size()-1; i++)
        a += wedge(poly[i]-poly[0], poly[i+1]-poly[0])/2;
    return a;
}
*/
