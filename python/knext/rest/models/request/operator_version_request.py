# coding: utf-8

"""
    knext

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""

#  Copyright 2023 Ant Group CO., Ltd.
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
#  in compliance with the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied.

import pprint
import re  # noqa: F401

import six

from knext.rest.configuration import Configuration


class OperatorVersionRequest(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'project_id': 'int',
        'operator_id': 'int'
    }

    attribute_map = {
        'project_id': 'projectId',
        'operator_id': 'operatorId'
    }

    def __init__(self, project_id=None, operator_id=None, local_vars_configuration=None):  # noqa: E501
        """OperatorVersionRequest - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._project_id = None
        self._operator_id = None
        self.discriminator = None

        self.project_id = project_id
        self.operator_id = operator_id

    @property
    def project_id(self):
        """Gets the project_id of this OperatorVersionRequest.  # noqa: E501


        :return: The project_id of this OperatorVersionRequest.  # noqa: E501
        :rtype: int
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id):
        """Sets the project_id of this OperatorVersionRequest.


        :param project_id: The project_id of this OperatorVersionRequest.  # noqa: E501
        :type: int
        """
        if self.local_vars_configuration.client_side_validation and project_id is None:  # noqa: E501
            raise ValueError("Invalid value for `project_id`, must not be `None`")  # noqa: E501

        self._project_id = project_id

    @property
    def operator_id(self):
        """Gets the operator_id of this OperatorVersionRequest.  # noqa: E501


        :return: The operator_id of this OperatorVersionRequest.  # noqa: E501
        :rtype: int
        """
        return self._operator_id

    @operator_id.setter
    def operator_id(self, operator_id):
        """Sets the operator_id of this OperatorVersionRequest.


        :param operator_id: The operator_id of this OperatorVersionRequest.  # noqa: E501
        :type: int
        """
        if self.local_vars_configuration.client_side_validation and operator_id is None:  # noqa: E501
            raise ValueError("Invalid value for `operator_id`, must not be `None`")  # noqa: E501

        self._operator_id = operator_id

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, OperatorVersionRequest):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, OperatorVersionRequest):
            return True

        return self.to_dict() != other.to_dict()