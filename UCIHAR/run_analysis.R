library(dplyr)
library(tidyr)

setwd("~/dropbox/coursera/UCI HAR Dataset")



#Appropriately labels the data set with descriptive variable names. 
activities <- read.table("activity_labels.txt", 
                         col.names=c("id", "activity_label"))
features <- read.table("features.txt", 
                       col.names=c("id", "feature"))

#Uses descriptive activity names to name the activities in the data set
get_activities <- function(filename) {
        read.table(filename, 
        	   col.names=c("activity_id")) %>%
        left_join(activities, by=c("activity_id"="id")) %>% 
        rename(id=activity_id)
}
activities_test <- get_activities("./test/y_test.txt")
activities_train <- get_activities("./train/y_train.txt")

# Extracts only the measurements on the mean and
# standard deviation for each measurement. 
get_features <- function(filename) {
        read.table(filename, 
                   col.names=features$feature, 
                   colClasses=c("numeric")) %>%
        select(contains("mean..", ignore.case=FALSE), 
               contains("std..", ignore.case=FALSE))
}
features_test <- get_features("./test/X_test.txt")
features_train <- get_features("./train/X_train.txt")

get_subjects <- function(filename) {
        read.table(filename, col.names=c("subject"))
}
subjects_test <- get_subjects("./test/subject_test.txt")
subjects_train <- get_subjects("./train/subject_train.txt")


test561 <- bind_cols(activities_test, 
		     subjects_test, 
		     features_test) %>%
        select(-id)
train561 <- bind_cols(activities_train, 
		      subjects_train, 
		      features_train) %>%
        select(-id)

#Merges the training and the test sets to create one data set.
five61 <- bind_rows(test561, train561)

feature_data <- five61 %>% gather(key=variable, 
                  value=measurement, 
                  -subject, -activity_label) %>%
        mutate(variable = gsub("\\.+", "_", variable)) %>%
        mutate(variable = sub("_$", "", variable)) %>%
        separate(col=variable, 
                 into=c("feature", "statistic", "axis"), 
                 sep="_", 
                 extra="drop") %>%
        mutate(feature = sub("Mag", "", feature)) %>%
        mutate(feature = gsub("(t|f)(Body|Gravity)(.*)", "\\1,\\2,\\3", feature)) %>%
        separate(feature, into=c("domain", "source", "measure"), sep=",") %>% 
        mutate(domain = ifelse(domain=="t", "time", "frequency"))

# From the data set in step 4, creates a second, independent tidy data set
# with the average of each variable for each activity and each subject.
summary_set <- feature_data %>%
        group_by(activity_label, subject, domain, source, measure, statistic, axis) %>%
        summarize(mean(measurement))

write.table(feature_data, "UCI_HAR_tidy.txt", row.name=FALSE)

write.table(summary_set, "UCI_HAR_Summary.txt", row.name=FALSE)
