from MyImage_class import *
from PIL import Image

class Classifier_LDA:

    def __init__(self):
        pass

    def Fisher_LDA_train(self, training_data):
        # training_data ---> a list whose every element is for a class --> every element of list ---> rows: samples, columns: dimensions
        # returns --> e_vecs (rows: eigenvectors, columns: dimensions of eigenvectors), e_vals (columns (elements): eigenvalues), projected_training_data --> a list whose elements are samples in rows and dimensions in columns
        number_of_classes = len(training_data)
        dimension_of_data = training_data[0].shape[1]
        # ----- find mean of classes:
        mean_of_classes = np.zeros((number_of_classes, dimension_of_data))
        for class_index in range(number_of_classes):
            mean_of_class = training_data[class_index].mean(axis=0)
            mean_of_classes[class_index, :] = mean_of_class
        # ----- within scatter:
        within_scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index in range(number_of_classes):
            number_of_samples_in_class = training_data[class_index].shape[0]
            for sample_index in range(number_of_samples_in_class):
                sample = training_data[class_index][sample_index,:]
                temp = (np.matrix(sample).T - np.matrix(mean_of_classes[class_index, :]).T) * (np.matrix(sample).T - np.matrix(mean_of_classes[class_index, :]).T).T
                temp = np.array(temp)
                within_scatter = within_scatter + temp
        # ----- between scatter:
        mean_of_means_of_classes = mean_of_classes.mean(axis=0)
        between_scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index in range(number_of_classes):
            temp = (np.matrix(mean_of_classes[class_index, :]).T - np.matrix(mean_of_means_of_classes).T) * (np.matrix(mean_of_classes[class_index, :]).T - np.matrix(mean_of_means_of_classes).T).T
            temp = np.array(temp)
            between_scatter = between_scatter + temp
        # ----- create the Fisher subspace:
        lambda_for_singularity = 0.0001
        e_vals, e_vecs = np.linalg.eigh( np.linalg.inv(within_scatter + lambda_for_singularity*np.eye(dimension_of_data)) * between_scatter )
        # for eigenvalue, see: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eigh.html
        # https://stackoverflow.com/questions/6684238/whats-the-fastest-way-to-find-eigenvalues-vectors-in-python
        # ----- sort the eigenvalues (and corresponding eigenvectors) from biggest to smallest:
        # https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
        idx = e_vals.argsort()[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:,idx]
        # ----- pick the number_of_classes-1 first eigenvectors and eigenvalues:
        number_of_valid_eigenvalues = min(number_of_classes-1, dimension_of_data)  # note: we have dimension_of_data eigenvalues from which we shouls pick
        e_vals = e_vals[:number_of_valid_eigenvalues]
        e_vecs = e_vecs[:number_of_valid_eigenvalues, :]
        # ----- project the training samples onto Fisher subspace:
        dimension_of_Fisher_subspace = e_vecs.shape[0]  # which is number_of_valid_eigenvalues
        projected_training_data = [np.empty((0, dimension_of_Fisher_subspace))] * number_of_classes
        for class_index in range(number_of_classes):
            number_of_samples_in_class = training_data[class_index].shape[0]
            for sample_index in range(number_of_samples_in_class):
                sample = training_data[class_index][sample_index,:]
                projected_sample = self.project_onto_Fisher_subspace(e_vecs=e_vecs, sample=sample)
                projected_training_data[class_index] = np.vstack([projected_training_data[class_index], projected_sample])
        # ----- returns:
        return e_vecs, e_vals, within_scatter, between_scatter, projected_training_data

    def Fisher_LDA_test(self, test_data, projected_training_data, e_vecs):
        # test_data ---> rows: samples, columns: dimensions
        # ---- find means of projected_training_data:
        number_of_classes = len(projected_training_data)
        dimension_of_Fisher_subspace = projected_training_data[0].shape[1]
        means_of_projected_training_samples = np.zeros((number_of_classes, dimension_of_Fisher_subspace))
        for class_index in range(number_of_classes):
            projected_training_class = projected_training_data[class_index]
            means_of_projected_training_samples[class_index, :] = projected_training_class.mean(axis=0)
        # ---- find class of each test sample:
        number_of_samples = test_data.shape[0]
        estimated_classes = np.zeros((number_of_samples, 1))
        for sample_index in range(number_of_samples):
            sample = test_data[sample_index, :]
            # project test sample onto Fisher subspace:
            projected_sample = self.project_onto_Fisher_subspace(e_vecs=e_vecs, sample=sample)
            # find the class of test sample:
            estimated_class_of_sample = self.find_closest_vector(test_vector=projected_sample, vectors=means_of_projected_training_samples)
            estimated_classes[sample_index, :] = estimated_class_of_sample
        return estimated_classes

    def project_onto_Fisher_subspace(self, e_vecs, sample):
        # sample --> a horizontal vector --> columns: dimensions
        # projected_sample --> a horizontal vector --> columns: dimensions
        dimension_of_Fisher_subspace = e_vecs.shape[0]
        projected_sample = np.zeros((1, dimension_of_Fisher_subspace))
        for eigen_index in range(dimension_of_Fisher_subspace):
            eigen_vector = e_vecs[eigen_index, :]
            projected_sample[0, eigen_index] = np.dot(sample, eigen_vector)  # inner dot
        return projected_sample

    def find_closest_vector(self, test_vector, vectors):
        # test_vector --> horizontal vector
        # vectors --> rows: vectors, columns: dimensions
        number_of_vectors = vectors.shape[0]
        min_distance = np.inf
        for vector_index in range(number_of_vectors):
            vector = vectors[vector_index]
            distance = self.Euclidean_distance(vector1=test_vector, vector2=vector)
            if distance < min_distance:
                min_distance = distance
                closest_vector_index = vector_index
        return closest_vector_index

    def Euclidean_distance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        distance = np.sqrt(sum((vector1 - vector2) ** 2))
        return distance