# Either run the script from the internal Blender script editor or execute the script on the console. If you are using an external editor, use the following commands on the blender console to run the script (this always takes the new version)
#
#import os 
#fname = "C:/Users/Jaswir Raghoe/Desktop/3dm/ddm3Package/ddm_prac3.py"
#exec(compile(open(fname).read(), fname, 'exec'))


import bpy
import math
import bmesh
import numpy as np
import scipy
from mathutils import *
import mathutils
import scipy.sparse.linalg as sla
from scipy.sparse import *
from numpy import *
from scipy import *
import time

# Runs the precalculation process, storing the precomputation data with the object that is being deformed
def Precompute(source_object):
    print("HOUSTON SPOTTED ACTIVITY IN PRECOMPUTE")
    # Get a BMesh representation
    bm = bmesh.new()              # create an empty BMesh
    bm.from_mesh(source_object.data)   # fill it in from a Mesh
    size = len(bm.verts)

    #Get const columns' indices
    CONSTindices = []
    for handle in get_handles(source_object):
        CONSTindices += handle[0]
    #Gets sorted + unique (no duplicates) list
    CONSTindices.sort()
    CONSTindices = list(set(CONSTindices))
    

    print("Const indices length: ", len(CONSTindices))
  
    #INSTEAD OF REMOVING CONSTRAINT COLUMSN AFTERWARDS
    #WE PREVENT ADDING THEM IN THE FIRST PLACE.
    #Therefore subtract CONST length from amountOfcolumns
    amountOfcolumns = size - len(CONSTindices)

    #In order to prevent deleting columns and its inefficiency we 
    #avoid adding these columns alltogether. 
    #The fact that indices of vertices are directly linked to columns in A
    #can be used for this.
    #By using a kd_tree which links vertices to indices
    #in combination with this validIndices list
    #we can map columns in A to their appropriate validIndices
    validIndices = list(set(range(0, size)) - set(CONSTindices))

    #Use KD-Tree to link vertices to indices in bm.verts
    verticesKD_Tree = mathutils.kdtree.KDTree(size)
    for i, v in enumerate(bm.verts):
        verticesKD_Tree.insert(v.co, i)

    verticesKD_Tree.balance()

    #Finds the total amount of rows
    amountOfRows = 0
    for index, vertex in enumerate(bm.verts):
            one_ringNeighbours = [edge.other_vert(vertex) for edge in vertex.link_edges]
            amountOfRows += len(one_ringNeighbours)

    #Stores APrime
    APrime = lil_matrix((amountOfRows, amountOfcolumns))

    #Store A (needed for computing b...)
    A = lil_matrix((amountOfRows , amountOfcolumns + len(CONSTindices)))
    #Used to count which row we are currently working at
    rowCount = 0
    #Used to store sqrt(Wiv)(xv-xi) (constant)
    sqrtWivTimesxvMinxi = []
    for index, vertex in enumerate(bm.verts):

        #Get oneRing neighbours of vertex
        one_ringNeighbours = [edge.other_vert(vertex) for edge in vertex.link_edges]
        print("Precomputing..")

        for neighbour in one_ringNeighbours:

            #Get oneRing neighbours of neighbour
            neighboursNeighbours = [edge.other_vert(neighbour) for edge in neighbour.link_edges]
            #Find matching neighbours
            matchingNeighbours =  list(set(one_ringNeighbours) & set(neighboursNeighbours))

            s = len(matchingNeighbours)
            
            
            #Set weight to 1 incase division by 0, or not enclosed one ring 
            #Resulting in only 1 angle instead of two
            Wiv = 1

            #Use these to compute weights, see https://in.answers.yahoo.com/question/index?qid=20110530115210AA2eAW1
            aAlpha = neighbour.co - matchingNeighbours[0].co
            bAlpha = vertex.co - matchingNeighbours[0].co
            aAlphacrossbAlphadotmagnitude = ((aAlpha.cross(bAlpha)).magnitude)
            aAlphadotbAlpha = aAlpha.dot(bAlpha)
            if not(aAlphadotbAlpha == 0):
                tanAlpha =  aAlphacrossbAlphadotmagnitude/aAlphadotbAlpha 
                cotAlpha = 1/tanAlpha
                cotBeta = 0 
                if s == 2:
                    aBeta  = neighbour.co - matchingNeighbours[1].co
                    bBeta  = vertex.co - matchingNeighbours[1].co
                    aBetacrossbBetadotmagnitude = aBeta.cross(bBeta).magnitude
                    aBetadotbBeta = aBeta.dot(bBeta)
                    if not(aBetadotbBeta == 0):
                        tanBeta  = aBetacrossbBetadotmagnitude / aBetadotbBeta
                        cotBeta  = 1/tanBeta 
                        Wiv = 0.5*(cotAlpha + cotBeta)

            #AMIR: Some people asked what to do with negative cotangent weight, 
            #because they commonly take sqrt(w_ij) to put into the ||Ax-b||^2 expression. 
            #The weights should not overly negative in reasonable triangles, 
            #so try and use w_{ij}={small positive epsilon} as a cheap workaround. For instance w_{ij}=10e-3.
            if (Wiv < 0):
                Wiv = 10e-3

            Wiv = math.sqrt(Wiv)

            #Fills APrime and A
            neighbourIdx = verticesKD_Tree.find(neighbour.co)[1]

            #For A no checks needed:
            A[rowCount , index] =   Wiv
            A[rowCount , neighbourIdx] = -Wiv

            #For APrime:
            #Check whether index is valid, if not no need to add it to A
            validNeighbour = neighbourIdx in validIndices
            validVertex = index in validIndices
            #If valid, get its appropriate index and fill A
            if validNeighbour:
                #Map index to real column:
                neighbourIdx = validIndices.index(neighbourIdx)
                APrime[rowCount, neighbourIdx] = -Wiv
            if validVertex:
                #Map index to real column:
                vertexIdx = validIndices.index(index)
                APrime[rowCount , vertexIdx] =  Wiv
          
            sqrtWivxv_xi = np.array(Wiv*(vertex.co - neighbour.co))
            sqrtWivTimesxvMinxi.append(sqrtWivxv_xi)

            #update row
            rowCount += 1
            
    #Stores oldA, for computing b'      
    oldA = A;

    #Computes A'TA' and prefactors the matrix
    APrime = csc_matrix(APrime)
    APrimeT = csc_matrix(APrime.T)
    APrimeTAPrime = csc_matrix(APrimeT * APrime)
    lu = sla.splu(APrimeTAPrime)

    #Reduces lu object to explicit representation of the decomposition
    L = lu.L.A
    U = lu.U.A
    n = APrimeTAPrime.shape[0]
    Pr = lil_matrix((n, n))
    Pr[lu.perm_r, np.arange(n)] = 1
    Pc = lil_matrix((n, n))
    Pc[np.arange(n), lu.perm_c] = 1

    #Stores: 
    #- lu decomposition
    #- A
    #- A'T
    #- amountOfrows
    #- sqrtWivTimesxvMinxi
    #to use further in the computation of ARAP.
    #Sparse matrices are Sparse Packaged (since sparse matrices can't be saved on
    #a carrying object normally).
    #This makes for efficiënt storing as only indices of non-zero elements are stored
    #See Package_SparseMatrix() for more details.
    Package_SparseMatrix(oldA, source_object, 'oldA')
    Package_SparseMatrix(APrimeT, source_object, 'APrimeT')
    Package_SparseMatrix(L, source_object, 'L')
    Package_SparseMatrix(U, source_object, 'U')
    Package_SparseMatrix(Pr, source_object, 'Pr')
    Package_SparseMatrix(Pc, source_object, 'Pc')

    source_object.data['sqrtWivTimesxvMinxi'] = sqrtWivTimesxvMinxi
    source_object.data['precomputed_data_amountOfRows'] = amountOfRows

# Runs As Rigid As Possible deformation on the mesh M, using a list of handles given by H. A handle is a list of vertices tupled with a transform matrix which might be rigid (identity)
def ARAP(source_mesh, deformed_mesh, H, existingDeformed, start):
    # Get a BMesh representation for source
    bm = bmesh.new()              # create an empty BMesh
    bm.from_mesh(source_mesh)   # fill it in from a Mesh

    #Computes Local Step 
    Rvs = []
    size = len(bm.verts)
    
    # If this is the initial guess, apply translation on constraint vertices of deformed_mesh
    if not(existingDeformed):
        for handle in H:
            #Use to find out whether a vertex is a constraint vertex and get initial guess if needed
            CONST = handle[0]
            translationMatrix = handle[1]
            for index, vertex in enumerate(deformed_mesh.vertices):
                if index in CONST:
                    vertex.co =   vertex.co * translationMatrix

    #After possible initial guess, changing the deformed mesh
    #Get a BMesh representation for deformed 
    dm = bmesh.new()  # create an empty BMesh
    dm.from_mesh(deformed_mesh)   # fill it in from a Mesh

    for (sourceVertex, deformedVertex) in zip(bm.verts, dm.verts):
        #Composes Matrices P and Q 
        one_ringNeighboursPDiff = [sourceVertex.co - edge.other_vert(sourceVertex).co for edge in sourceVertex.link_edges]
        one_ringNeighboursQDiff = [deformedVertex.co - edge.other_vert(deformedVertex).co for edge in deformedVertex.link_edges]
    
        p = one_ringNeighboursPDiff
        pT = np.transpose(p)
        Q = one_ringNeighboursQDiff

        #Compose correlation matrix S3×3 = P^T Q.
        S3x3 = np.dot(pT, Q)  

        #Decompose S = UΣvT using singular value decomposition, 
        #and composes the closest rigid transformation Rv = UvT
        U, sigma, vT = np.linalg.svd(S3x3)
        Rv = np.dot(U, vT)

        #If det(Rv) = −1 (determinant), leading to reflection, instead compute Rv = UΣvT
        # where Σ'is an identity matrix, save for Σ'ii = -1 , where i is the index of 
        #the smallest diagonal (singular) value in the original Σ. . 
        #(flipping sign). For instance, if i = 3, you should use Σ' = diag[1, 1, −1].
        detRv = np.linalg.det(Rv)
        if(detRv == -1):
            #Find the index of the smallest diagonal vlaue in the original Σ.
            i =  np.argmin(sigma)
            #Computes Σ'
            s = [1,1,1]
            s[i] = -1
            sigmaTag = np.array([[s[0],0,0], [0,s[1],0], [0,0,s[2]]])
            #Computes reflection resistant  Rv
            Rv = np.dot(np.dot(U, sigmaTag), vT)
        Rvs.append(Rv)


    #Sets up b
    amountOfcolumns = 3
    amountOfRows = source_mesh['precomputed_data_amountOfRows']

    #Stores b
    b = lil_matrix((amountOfRows, amountOfcolumns))

    #Used to count which row we are currently working at
    rowCount = 0

    #b'= b - A (   0   ) <= last thing we call ZeroXPrimeConst
    #          (X'Const)
    #Fills b and ZeroXPrimeConst
    ZeroXPrimeConst = lil_matrix((size ,3))
    sqrtWivTimesxvMinxi = source_mesh['sqrtWivTimesxvMinxi']

    CONSTindices = []
    for handle in H:
        CONSTindices += handle[0]

    for index, vertex in enumerate(dm.verts):

        #If vertex is a constraint index, the proper coördinates should be inserted in the matrix
        if index in CONSTindices:
            ZeroXPrimeConst[index] = np.array(vertex.co)
        #Otherwise insert zero's for coördinates 
        else:
            ZeroXPrimeConst[index] = np.zeros((3))

        #Get oneRing neighbours of vertex
        one_ringNeighbours = [edge.other_vert(vertex) for edge in vertex.link_edges]

        for neighbour in one_ringNeighbours:

            #RowInput => sqrt(Wiv)(xv-xi)Rv
            sqrtWivxv_xi =  sqrtWivTimesxvMinxi[rowCount] 
            Rv =  Rvs[index]
            rowInput  = np.dot(sqrtWivxv_xi , Rv)
            b[rowCount] =  rowInput

            #update row
            rowCount += 1



    #b'= b - A (   0   )
    #          (X'Const)
    ZeroXPrimeConst = csc_matrix(ZeroXPrimeConst)
    b = csc_matrix(b)

    print("ARAP..")
    

	
    A = UnPackage_SparsePackage(source_mesh, 'oldA')

  
    bPrime = b - (A * ZeroXPrimeConst)

    #Retrieve the component to compose A'TA', and recompose A'TA'
    L = UnPackage_SparsePackage(source_mesh, 'L')
    U = UnPackage_SparsePackage(source_mesh, 'U')
    Pr = UnPackage_SparsePackage(source_mesh, 'Pr')
    Pc = UnPackage_SparsePackage(source_mesh, 'Pc')
    APrimeTAPrime = csc_matrix((Pr.T * (L* U) * Pc.T).A)

    #Retrieve A'T necessary to multiply with bPrime
    APrimeT = UnPackage_SparsePackage(source_mesh, 'APrimeT')
    APrimeTxbPrime = csc_matrix(APrimeT*bPrime)

    #Computes xPrime
    xPrime = sla.spsolve(APrimeTAPrime, APrimeTxbPrime)

    #Puts x' vertices in deformed_mesh for next iteration
    xPrimeidx = 0
    for index, vertex in enumerate(deformed_mesh.vertices):
        #Only fill vertices that are not constraint
        if not(index in CONSTindices):
            vertex.co = mathutils.Vector((xPrime[xPrimeidx, 0],
                                          xPrime[xPrimeidx, 1],
                                          xPrime[xPrimeidx, 2]))
            xPrimeidx += 1

    #Such that the maximum absolute movement of the vertices between this mesh and the one in the next iteration can be found
    #which can then be compared with tolerance to decide if ARAP is done
    return deformed_mesh

#Used for visualisation purposes (e.g points as spheres)
def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    return mat

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

#SparsePackaging: given a sparseMatrix, the bpyObject to carry relevant data and a key
#decomposes matrix into shape and decomposition: rowIndices, columnIndices and values 
#at respective row,column positions.
def Package_SparseMatrix(sparseMatrix, bpyObject, stringKey):
    bpyObject.data[stringKey + 'SparsePackageShape'] = tuple(sparseMatrix.shape)
    bpyObject.data[stringKey + 'SparsePackageDecomposition'] = np.array(find(sparseMatrix))

#Returns an unpackaged sparse package, given 
#the bpyObject carrying the data and key of the string.
def UnPackage_SparsePackage(bpyObject, stringKey):
    packageShape = tuple(bpyObject[stringKey + 'SparsePackageShape'])
    rowIndices, columnIndices, values =  bpyObject[stringKey + 'SparsePackageDecomposition']

    sparseMatrix = lil_matrix(packageShape)
    for (rowIndex, columnIndex, value) in zip(rowIndices, columnIndices, values):
        sparseMatrix[rowIndex, columnIndex] = value
    return csc_matrix(sparseMatrix)


def getHandleNames():
    # Only search up to (and not including) this number of handles
    max_handles = 10
    handleNames = []
    handleDestinationNames = []
    handleTransforms = []
    destinationTransforms = []
    # For all numbered handles
    for i in range(max_handles):

        # Construct the handles representative name
        handle_name = 'handle_' + str(i)

        if bpy.data.objects.get(handle_name) is not None:
            handleNames.append(handle_name)
            handleTransforms.append(get_transform_of_object(handle_name))

             # If a destination box exists
            handle_dest_name = handle_name + '_dest'
            if bpy.data.objects.get(handle_dest_name) is not None:
                handleDestinationNames.append(handle_dest_name)
                destinationTransforms.append(get_transform_of_object(handle_dest_name))

    return[handleNames , handleDestinationNames, handleTransforms, destinationTransforms]

#Checks whether handles have been added/removed/changed
#or whether no previousHandles which must mean precompute
def precomputeIsDirty(source):
    #Check whether handles have been added/removed/changed:
    handles = getHandleNames()
    previousHandles = None
    if 'previousHandles' in source.data:
        PackedpreviousHandles = source.data['previousHandles']
        #Unpack from bpy structure
        unpackedpreviousHandles0 = list(PackedpreviousHandles[0])
        unpackedpreviousHandles1 = list(PackedpreviousHandles[1])
        unpackedpreviousHandles2 = [Matrix(item) for item in PackedpreviousHandles[2]]
        unpackedpreviousHandles3 = [Matrix(item) for item in PackedpreviousHandles[3]]

        previousHandles = [unpackedpreviousHandles0 , unpackedpreviousHandles1, unpackedpreviousHandles2, unpackedpreviousHandles3]

        #If handles got added or removed, change previousHandles to current
        #set handles have changed!
        if(not(handles == previousHandles)):
            source.data['previousHandles'] = handles

    else:
        #There were no handle names store before, so must precompute
        source.data['previousHandles']  = handles

    return not(handles == previousHandles)

def main():

    #Used to time computation speed on 1 or more checkpoints
    start = time.time()
    source = bpy.data.objects['source']


    #Precomputes A'^T * A if the data is dirty or does not exist, and stores its decomposition,
    #amountOfrows in A' , 
    #sqrtWivTimesxvMinxi and,
    #A'T (both used to compute b in global step, but this is constant so can be computed once(assuming clean data))
    #along with with the source object.
    DataisDirty = precomputeIsDirty(source)
    data = ['oldASparsePackageShape', 'oldASparsePackageDecomposition', 'APrimeTSparsePackageShape'
    , 'APrimeTSparsePackageDecomposition', 'LSparsePackageShape', 'LSparsePackageDecomposition', 
    'USparsePackageDecomposition', 'USparsePackageShape', 'PrSparsePackageShape', 'PrSparsePackageDecomposition' 
    ,'PcSparsePackageShape', 'PcSparsePackageDecomposition', 'sqrtWivTimesxvMinxi'
    , 'precomputed_data_amountOfRows']
    dataExists =  all([dataItem in source.data for dataItem in data])




    print("Is the data dirty?: {0}" .format(DataisDirty))
    print("Does the data exist?: {0}" .format(dataExists))
    if(DataisDirty or not(dataExists)):
        Precompute(source)

    print('It took {0:0.1f} seconds to complete Precompute'.format(time.time() - start))



    #Perform As Rigid As Possible deformation on the source object in the first iteration, and on a deformed object if it exists
    deformed = None
    existingDeformed = bpy.data.objects.get("deformed") is not None
    #Checks for an existing deformed mesh
    if existingDeformed:
        #If there exists one already, use that as the deformed_mesh for ARAP
        deformed = bpy.data.objects['deformed'].data
    else:
        #Otherwise make an initial guess
        deformed = get_deformed_object(source).data


    # Rotate deformed to source default rotation
    bpy.data.objects["deformed"].rotation_euler[0] = 3.14159 * 90 / 180;


    #Gets the diagonal of the bounding box of source
    #and take 10^-4 of that as a tolerance for absolute mesh movement between iterations
    name = "source"

    origin = mathutils.Vector((0,0,0))
    for index, point in enumerate(source.data.vertices):
         origin += point.co

    origin /= len(source.data.vertices)

    xDimensionsHalve = bpy.data.objects[name].dimensions.x/2  
    xMin = 0 - xDimensionsHalve
    xMax = 0 + xDimensionsHalve

    yDimensionsHalve = bpy.data.objects[name].dimensions.y/2
    yMin = 0 - yDimensionsHalve
    yMax = 0 + yDimensionsHalve

    zDimensionsHalve = bpy.data.objects[name].dimensions.z/2
    zMin = 0 - zDimensionsHalve
    zMax = 0 + zDimensionsHalve

    v1 =  np.array((xMin, yMin,zMax))
    v2 =  np.array((xMax, yMax,zMin))
    diagonalOfBoundingBox = np.linalg.norm(v1-v2)

    tolerance = 10e-5 * diagonalOfBoundingBox


    #Iterate until the maximum absolute movement between two iteration is smaller than tolerance
    previousIteration = ARAP(source.data, deformed, get_handles(source), existingDeformed, start)
    maximumAbsoluteMovementOfVertices = 2 * tolerance
    iteration = 2
    while not(maximumAbsoluteMovementOfVertices < tolerance) :
        existingDeformed = True
        deformed = bpy.data.objects['deformed'].data
        currentIteration = ARAP(source.data, deformed, get_handles(source), existingDeformed, start)
        #Finds maximum absolute movement of all verticesbetween iterations
        absoluteMovements = []
        for (previousIterationVertex, currentIterationVertex) in zip(previousIteration.vertices, currentIteration.vertices):
            absoluteMovement = np.linalg.norm(np.array(previousIterationVertex.co) - np.array(currentIterationVertex.co))
            absoluteMovements.append(absoluteMovement)
        maximumAbsoluteMovementOfVertices = max(absoluteMovements)

        print('It took {0:0.1f} seconds to complete iteration {1} of ARAP'.format((time.time() - start) , iteration))
        print("The max absolute movement of vertices between iterations was: {0}" .format(maximumAbsoluteMovementOfVertices))
        print("The tolerance is: {0}" .format(tolerance))
        print("maximumAbsoluteMovementOfVertices < tolerance = {0}" .format(maximumAbsoluteMovementOfVertices < tolerance))

        #prepares for next iteration
        previousIteration = currentIteration
        iteration += 1

# BLENDER
# -------
    
def get_transform_of_object(name):
    return bpy.data.objects[name].matrix_basis

def get_mesh_vertices(name):
    return bpy.data.objects[name].data.vertices
    
# Finds the relative transform from matrix M to T
def get_relative_transform(M, T):
    
    Minv = M.copy()
    Minv.invert()
        
    return T * Minv

# Returns an object that can be used to store the deformed mesh
def get_deformed_object(source):
    
    name = 'deformed'
    
    # Create an object if it doesn't yet exist
    if bpy.data.objects.get(name) is None:
    
        # Create new mesh
        mesh = bpy.data.meshes.new(name)
     
        # Create new object associated with the mesh
        ob_new = bpy.data.objects.new(name, mesh)
     
        scn = bpy.context.scene
        scn.objects.link(ob_new)
        scn.objects.active = ob_new
     
        # Copy data block from the old object into the new object
        ob_new.data = source.data.copy()
        ob_new.scale = source.scale
        ob_new.location = source.location
    
    return bpy.data.objects[name]
    
# Find the vertices within the bounding box by transforming them into the bounding box's local space and then checking on axis aligned bounds.
def get_handle_vertices(vertices, bounding_box_transform, mesh_transform):

    result = []

    # Fibd the transform into the bounding box's local space
    bounding_box_transform_inv = bounding_box_transform.copy()
    bounding_box_transform_inv.invert()
    
    # For each vertex, transform it to world space then to the bounding box local space and check if it is within the canonical cube x,y,z = [-1, 1]
    for i in range(len(vertices)):
        vprime = vertices[i].co.copy()
        vprime.resize_4d()
        vprime = bounding_box_transform_inv * mesh_transform * vprime
        
        x = vprime[0]
        y = vprime[1]
        z = vprime[2]
        
        if (-1 <= x) and (x <= 1) and (-1 <= y) and (y <= 1) and (-1 <= z) and (z <= 1):
            result.append(i)

    return result

# Returns a list of handles and their transforms
def get_handles(source):
    
    result = []
    
    mesh_transform = get_transform_of_object(source.name)
    
    # Only search up to (and not including) this number of handles
    max_handles = 10
    
    # For all numbered handles
    for i in range(max_handles):
    
        # Construct the handles representative name
        handle_name = 'handle_' + str(i)
        
        # If such a handle exists
        if bpy.data.objects.get(handle_name) is not None:
            
            # Find the extends of the aligned bounding box
            bounding_box_transform = get_transform_of_object(handle_name)
            
            # Interpret the transform as a bounding box for selecting the handles
            handle_vertices = get_handle_vertices(source.data.vertices, bounding_box_transform, mesh_transform)
            
            # If a destination box exists
            handle_dest_name = handle_name + '_dest'
            if bpy.data.objects.get(handle_dest_name) is not None:
                
                bounding_box_dest_transform = get_transform_of_object(handle_dest_name)
                
                result.append( (handle_vertices, get_relative_transform(bounding_box_transform, bounding_box_dest_transform) ) ) 
                
            else:
            
                # It is a rigid handle
                m = Matrix()
                m.identity()
                result.append( (handle_vertices, m) )
            
    return result
    
main()