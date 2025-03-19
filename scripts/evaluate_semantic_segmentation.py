
import numpy as np
import json
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import pandas as pd
from pathlib import Path
# import torch
from natsort import natsorted

import sys
sys.path.append("")
from scripts.utils_data import create_skeleton_gt_data
from scripts.utils_skeletonisation import load_json


# class runningScore(object):
# 	def __init__(self, n_classes):
# 		self.n_classes = n_classes
# 		self.confusion_matrix = np.zeros((n_classes, n_classes))

# 	def _fast_hist(self, label_true, label_pred, n_class):
# 		"""label_true =array of size [batch_szie*img_w*img_h
# 			label_false =array of size [batch_szie*img_w*img_h
			
# 			using bincount count number of pixels 
			
# 			returns hist, with for example hist[0][0], number of ((label_true==0) & (label_pred==0))
# 			and hist[0][1] ((label_true==0) & (label_pred==1))
# 			"""
# 		mask = (label_true >= 0) & (label_true < n_class)
# 		hist = np.bincount(
# 			n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
# 		).reshape(n_class, n_class)
# 		return hist

# 	def update(self, label_trues, label_preds):
# 		# for lt, lp in zip(label_trues, label_preds):
# 		#     self.confusion_matrix  +=  self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
# 		self.confusion_matrix += self._fast_hist(label_trues, label_preds, self.n_classes)

# 	def get_scores(self):
# 		"""Returns accuracy score evaluation result.
# 			- overall accuracy
# 			- mean accuracy
# 			- mean IU
# 			- fwavacc
# 		"""
# 		hist = self.confusion_matrix
# 		acc = np.diag(hist).sum() / hist.sum()
# 		acc_cls = np.diag(hist) / hist.sum(axis=1)
# 		acc_cls = np.nanmean(acc_cls)
# 		iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) ## basiscally calculating correct_pixels_per_class / non_correctpixels_per_class
# 		mean_iu = np.nanmean(iu)
# 		freq = hist.sum(axis=1) / hist.sum()
# 		fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
# 		cls_iu = dict(zip(range(self.n_classes), iu))

# 		return (
# 			{
# 				"Overall Acc: \t": acc,
# 				"Mean Acc : \t": acc_cls,
# 				"FreqW Acc : \t": fwavacc,
# 				"Mean IoU : \t": mean_iu,
# 			},
# 			cls_iu,
# 		)

# 	def reset(self):
# 		self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def calculate_score(y_true, y_pred, labels, ignore_label=255, save_name=None):
	# safety check to check y_true and labels
	# if not sorted(np.unique(y_true)) == sorted(labels):
	#     print(np.unique(y_true))
	#     print(labels)
	#     print("ERROR please check labels")
	#     exit()
	# calculate score per class
	# ignore all predictions with ignore_label
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	bool_array = np.array(y_true)!=ignore_label
	y_true=y_true[bool_array]
	y_pred=y_pred[bool_array]


	result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
	p, r, f = result[0], result[1], result[2]
	n_points = result[3]

	iou_class = jaccard_score(y_true, y_pred, average=None, labels=labels)
	iou_class_micro = jaccard_score(y_true, y_pred, average="micro", labels=labels)
	iou_class_macro = jaccard_score(y_true, y_pred, average="macro", labels=labels)
	## iou_micro 
	# TP = (y_true==y_pred).sum()
	# FP_FN = (y_true!=y_pred).sum()
	# temp_iou_class_micro = TP / len(y_true) # is equal to jaccard_score(y_true, y_pred, average="micro", labels=labels), NEXT IS WRONG DONT -> jaccard_score(y_true, y_pred, average="micro")
	# iou_class_leaves
	# TP_LEAVES = (y_true[y_true==1]==y_pred[y_true==1]).sum()
	# TP_LEAVES/ (len(y_true[y_true==1])+(y_pred[y_true!=1]==1).sum())

	df = pd.DataFrame(
		np.array([labels, p, r, f, iou_class]).T,
		columns=["classes", "precision", "recall", "f-score", "iou_class"],
	)
	df["iou_micro"] = iou_class_micro
	df["iou_macro"] = iou_class_macro
	
	if save_name is not None:
		df.to_csv(str(save_name))

	# print("precision is tp/(tp+fp), how many were correct")
	# print("recall is tp/(tp+fn), how many did you miss")
	print("fscore", df["f-score"].mean())
	print("IoU_macro", iou_class_macro)
	return df

class EvaluationSemantic():
	
	def __init__(self, 
			  dt_graph_dir: Path, 
				gt_json=None,

			  cfg={"json_name": "test"}):
		
		# self.gt_path_dir = gt_path_dir
		self.dt_path_dir = dt_graph_dir
		self.gt_json = gt_json
		# self.cfg = cfg
		# self.split = cfg["json_name"]
		self.save_folder = Path("./Resources/")

	def load_all_gt_filenames(self):
		file_names = load_json(self.gt_json)
		return natsorted([self.gt_json.parent / f["sem_seg_file_name"] for f in file_names])

	def evaluate_pairs(self, labels=[1,2,3,4]):
		y_true_all, y_pred_all = [], []

		# obj = runningScore(n_classes=4)
		df_per_plant = pd.DataFrame()
		for gt_name in self.load_all_gt_filenames():
			plant_id = gt_name.stem.replace("_labels","")

			y_true = self.load_gt_data(gt_name)
			# print(np.unique(y_true))

			y_pred = pd.read_csv(self.dt_path_dir / (plant_id + ".txt"))["class_pred"].values + 1 # +1 because
			assert not len(y_true) != len(y_pred), "size not correct"
				
			# y_true, y_pred, labels = load_data(gt_name, pred_folder)
			y_true_all += y_true.tolist()
			y_pred_all += y_pred.tolist()

			df_temp = calculate_score(y_true=y_true.tolist(), y_pred=y_pred.tolist(), labels=labels)
			df_temp["plant_nr"] = plant_id
			df_per_plant = pd.concat([df_per_plant, df_temp])

		df_all = calculate_score(y_true=y_true_all, y_pred=y_pred_all, labels=labels)
		print("average metrics for all plants:")
		print(df_all)
		
		df_all_save_name =  self.save_folder / "metrics_semantic_segmenation.csv"
		df_all.to_csv(df_all_save_name, index=False)
		print(f"Average metrics: {df_all_save_name}")

		df_per_plant_name = self.save_folder / "metrics_semantic_segmenation_per_plant.csv"
		df_per_plant.to_csv(df_per_plant_name, index=False)
		print(f"Single plant metrics saved: {df_per_plant_name}")

		return df_all, df_per_plant


	def load_gt_data(self, gt_name):
		print("Loading", gt_name)
		df_semantic = pd.read_csv(str(gt_name))
		return df_semantic["semantic"].values
	
if __name__=="__main__":
	
	gt_json = Path(r"W:\PROJECTS\VisionRoboticsData\GARdata\datasets\tomato_plant_segmentation\20241223_csv_skeletons\anns\0-paper-2Dto3D\json\test.json")
	dt_graph_dir = Path("./Resources/output_semantic_segmentation")
	obj = EvaluationSemantic(dt_graph_dir=dt_graph_dir, gt_json=gt_json)
	obj.evaluate_pairs()
